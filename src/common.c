/*
 * common.c — definitions of the shared Blink runtime (declared in common.h).
 *
 * Compiled once into the `blink_common` library and linked into every
 * benchmark, rather than being a header full of `static inline` functions
 * instantiated in each of the ~40 translation units.  Besides being cleaner and
 * faster to build, this gives accurate gcov coverage: a single .gcda for the
 * shared code instead of per-TU copies that gcov cannot merge.
 */
#include "common.h"

/* ── globals ─────────────────────────────────────────────────────────────── */
int    my_rank;
int    w_size;
int    master_rank     = 0;
long long curr_iters;       /* total outer iterations executed (warmup + measured) */
long long measured_iters;   /* number of recorded samples (excludes warmup) */
int    warm_up_iters   = 5;
int    max_samples     = 1000;
double *durations;

/* Async-signal-safe shutdown flag set by sig_handler; checked in measurement
 * loops. The signal handler does no MPI / no malloc / no stdio — only sets
 * this flag. The main thread observes it between iterations and exits the
 * loop cleanly, then calls write_results() + MPI_Finalize() itself.        */
volatile sig_atomic_t shutdown_requested = 0;

/* common benchmark parameters – initialised to defaults, set by parse_common_args() */
int    msg_size            = 1024;
int    measure_granularity = 1;
int    rand_seed           = 1;
int    max_iters           = 1;
int    endless             = 0;
double burst_length        = 0.0;
int    burst_length_rand   = 0;   /* set by -bldist */
double burst_pause         = 0.0;
int    burst_pause_rand    = 0;   /* set by -bpdist */
int    pretty_output       = 0;
int    debug_mode          = 0;
int    use_gpu             = 0;

/* runtime-distribution plot (-plot). */
int    plot_output     = 0;
char   plot_stat[16]   = "max";   /* avg | min | max | median | mainrank */
int    plot_bins       = 10;
double plot_bin_size   = 0.0;     /* 0 => use plot_bins                  */
int    plot_log        = 0;

/* distribution selection for burst length and pause (-bldist / -bpdist). */
char   burst_dist[16]  = "exp";
double burst_shape     = 1.5;
char   pause_dist[16]  = "exp";
double pause_shape     = 1.5;

/* Weak default — benchmarks that take their own flags override this with a
 * strong definition at file scope (a one-liner `const char *benchmark_help =
 * "..."` in the .c file).  Benchmarks with no custom flags need do nothing. */
__attribute__((weak)) const char *benchmark_help = NULL;

/* GPU-mode capability flags.  Overridden by strong definitions in the few executables
 * that differ (Buffer-less ones set benchmark_has_data_buffers = 0, the one-sided
 * incast benchmarks set benchmark_uses_rma = 1).                               */
__attribute__((weak)) int benchmark_has_data_buffers = 1;
__attribute__((weak)) int benchmark_uses_rma         = 0;

#ifndef BLINK_PLOT_MAX_BINS
#define BLINK_PLOT_MAX_BINS 64
#endif
#define BLINK_PLOT_BAR_MAX  40
#define ALIGNMENT (sysconf(_SC_PAGESIZE))

/* ── random duration samplers ────────────────────────────────────────────── */

/* Exponential: shape parameter accepted for API uniformity but not used —
 * the distribution is fully determined by its mean.                         */
double rand_exp(double mean, double shape)
{
    (void)shape;
    double u = (rand() + 0.5) / (RAND_MAX + 1.0); /* shift away from 0 and 1 */
    return -mean * log(1.0 - u);
}

/* Pareto: shape = tail exponent α.  Must be > 1 for a finite mean.
 * Parameterised so that E[X] = mean regardless of α.
 * Inverse-CDF method: x = x_m * u^(-1/α), u ~ Uniform(0,1).               */
double rand_pareto(double mean, double shape)
{
    double alpha = shape;
    if (alpha <= 1.0) {
        fprintf(stderr, "rand_pareto: shape (alpha) must be > 1, got %g\n", alpha);
        exit(-1);
    }
    if (mean <= 0.0) {
        fprintf(stderr, "rand_pareto: mean must be > 0, got %g\n", mean);
        exit(-1);
    }
    double x_m = mean * (alpha - 1.0) / alpha;
    double u   = (rand() + 0.5) / (RAND_MAX + 1.0); /* shift away from 0 */
    return x_m * pow(u, -1.0 / alpha);
}

/* Log-normal: shape = σ of the underlying normal.
 * Parameterised so that E[X] = mean regardless of σ.
 * Box-Muller transform.                                                      */
double rand_lognormal(double mean, double sigma)
{
    if (mean <= 0.0) {
        fprintf(stderr, "rand_lognormal: mean must be > 0, got %g\n", mean);
        exit(-1);
    }
    if (sigma <= 0.0) {
        fprintf(stderr, "rand_lognormal: sigma must be > 0, got %g\n", sigma);
        exit(-1);
    }
    double mu = log(mean) - 0.5 * sigma * sigma;
    double u1 = (rand() + 0.5) / (RAND_MAX + 1.0); /* shift away from 0 */
    double u2 = (rand() + 0.5) / (RAND_MAX + 1.0);
    double z  = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
    return exp(mu + sigma * z);
}

/* Dispatch to the selected sampler. */
double rand_duration(double mean, const char *dist, double shape)
{
    if (strcmp(dist, "pareto")    == 0) return rand_pareto(mean, shape);
    if (strcmp(dist, "lognormal") == 0) return rand_lognormal(mean, shape);
    return rand_exp(mean, shape);
}

/*sleep seconds given as double*/
int dsleep(double t)
{
    struct timespec t1, t2;
    t1.tv_sec = (long)t;
    t1.tv_nsec = (t - t1.tv_sec) * 1000000000L;
    return nanosleep(&t1, &t2);
}

/*double comparison function for quicksort*/
int compare_doubles(const void *p1, const void *p2)
{
    if (*(double *)p1 < *(double *)p2)
        return -1;
    else if (*(double *)p1 > *(double *)p2)
        return 1;
    else
        return 0;
}

/* Bounds-checked accessor for a flag's value argument.  Aborts with a clear
 * message instead of dereferencing argv[argc] (== NULL) when a value-taking
 * flag is supplied as the final command-line token.                         */
const char *arg_value(int argc, char **argv, int *i)
{
    if (*i + 1 >= argc) {
        if (my_rank == master_rank)
            fprintf(stderr, "Missing value for option %s\n", argv[*i]);
        MPI_Abort(MPI_COMM_WORLD, -1);
    }
    return argv[++(*i)];
}

/* Validate a distribution name / shape pair selected via -bldist/-bpdist so a
 * misconfiguration fails loudly at startup rather than silently producing
 * garbage durations later (e.g. log-normal with a non-positive mean).        */
void validate_burst_dist(const char *what, const char *dist,
                         double mean, double shape)
{
    if (mean <= 0.0) {
        if (my_rank == master_rank)
            fprintf(stderr, "%s: randomised distribution requires a positive mean, got %g\n",
                    what, mean);
        MPI_Abort(MPI_COMM_WORLD, -1);
    }
    if (strcmp(dist, "pareto") == 0 && shape <= 1.0) {
        if (my_rank == master_rank)
            fprintf(stderr, "%s: pareto shape (alpha) must be > 1, got %g\n", what, shape);
        MPI_Abort(MPI_COMM_WORLD, -1);
    }
    if (strcmp(dist, "lognormal") == 0 && shape <= 0.0) {
        if (my_rank == master_rank)
            fprintf(stderr, "%s: lognormal shape (sigma) must be > 0, got %g\n", what, shape);
        MPI_Abort(MPI_COMM_WORLD, -1);
    }
}

/* Print the help block: common options + benchmark_help (if the benchmark
 * provides one).  Only the master rank prints, to avoid w_size duplicated
 * copies on stdout.  Callers handle MPI_Finalize + exit afterwards.          */
void print_help(const char *progname)
{
    if (my_rank != 0) return;
    const char *base = progname ? progname : "benchmark";
    const char *slash = strrchr(base, '/');
    if (slash) base = slash + 1;
    fprintf(stdout,
"Usage: %s [options]\n"
"\n"
"Common options:\n"
"  -mrank <int>                  master / root rank (default 0)\n"
"  -mrand                        choose master rank at random (uses -seed)\n"
"  -msgsize <bytes>              message size in bytes (default 1024)\n"
"  -iter <int>                   number of measured iterations (default 1)\n"
"  -warmup <int>                 number of warm-up iterations (default 5)\n"
"  -endl                         run endlessly until SIGUSR1\n"
"  -seed <int>                   RNG seed shared across ranks (default 1)\n"
"  -grty <int>                   ops per timed window (granularity, default 1)\n"
"  -maxsamples <int>             cap on per-rank samples stored (default 1000)\n"
"  -debug                        print per-rank DEBUG lines instead of timings\n"
"  -gpu                          keep Data buffers in GPU memory (needs a GPU-aware MPI)\n"
"  -pretty-print                 human-readable output (default: CSV row)\n"
"\n"
"Burst / pause (idle gaps between bursts of communication):\n"
"  -blength <duration>           burst length (default 0 = no bursting)\n"
"  -bpause  <duration>           pause length between bursts (default 0)\n"
"  -bldist  <exp|pareto|lognormal>   randomise burst length (default: fixed)\n"
"  -bpdist  <exp|pareto|lognormal>   randomise pause length (default: fixed)\n"
"  -blshape <double>             shape param for burst dist (pareto alpha / lognormal sigma)\n"
"  -bpshape <double>             shape param for pause dist\n"
"     (durations: bare number = seconds; suffixes 's', 'ms', 'us', 'ns' accepted)\n"
"\n"
"Per-iteration latency histogram (rank 0 only, end of run):\n"
"  -plot                         emit ASCII histogram of per-iteration latencies\n"
"  -plotstat <max|min|avg|median|mainrank>   cross-rank statistic (default: max)\n"
"  -plotbins <int>               number of bins (default 10, max 64)\n"
"  -plotbinsize <duration>       bin width (e.g. 2ms, 500us, 0.001) — overrides -plotbins\n"
"  -plotlog                      logarithmic bar heights\n"
"\n"
"  -h, --help                    show this help and exit\n",
        base);

    if (benchmark_help && *benchmark_help) {
        fprintf(stdout, "\nBenchmark-specific options:\n%s", benchmark_help);
    }
    fflush(stdout);
}

/* Wrap parse_duration with diagnostics + validation for CLI args that accept
 * a duration.  A bare number is seconds; suffixes s/ms/us/ns are recognised.
 * Aborts with a clear message on malformed input or negative value (0 is OK
 * because -blength 0 / -bpause 0 mean "no bursting").                       */
static double parse_duration_arg(const char *flag, const char *s)
{
    double v = parse_duration(s);
    if (isnan(v) || v < 0.0) {
        if (my_rank == master_rank)
            fprintf(stderr, "%s must be a non-negative duration "
                    "(e.g. 0.001, 1ms, 500us, 100ns), got '%s'\n", flag, s);
        MPI_Abort(MPI_COMM_WORLD, -1);
    }
    return v;
}

/* Parse a duration: a number with an optional unit suffix (s, ms, us, ns).
 * A bare number is interpreted as seconds (consistent with -blength/-bpause).
 * Returns the value in seconds, or NAN on a malformed string / unknown unit. */
double parse_duration(const char *s)
{
    char *end = NULL;
    double v = strtod(s, &end);
    if (end == s) return NAN;                 /* no leading number       */
    while (*end == ' ') end++;
    if (*end == '\0')           return v;     /* bare number => seconds  */
    if (strcmp(end, "s")  == 0) return v;
    if (strcmp(end, "ms") == 0) return v * 1e-3;
    if (strcmp(end, "us") == 0) return v * 1e-6;
    if (strcmp(end, "ns") == 0) return v * 1e-9;
    return NAN;                                /* unknown unit            */
}

/*
 * Parse the standard set of command-line flags shared by every benchmark.
 * Unrecognised flags are compacted to the front of argv (after argv[0]) and
 * the new argc is returned, so each benchmark can do a second pass for its
 * own flags.  Also seeds the RNG and resolves -mrand after parsing.
 */
int parse_common_args(int argc, char **argv)
{
    int new_argc       = 1; /* always keep argv[0] (program name) */
    int do_rand_master = 0;
    int i;

    for (i = 1; i < argc; i++) {
        if      (strcmp(argv[i], "-h")            == 0
              || strcmp(argv[i], "-help")         == 0
              || strcmp(argv[i], "--help")        == 0) {
            print_help(argv[0]);
            MPI_Finalize();
            exit(0);
        }
        else if (strcmp(argv[i], "-mrank")        == 0) { master_rank         = atoi(arg_value(argc, argv, &i)); }
        else if (strcmp(argv[i], "-mrand")        == 0) { do_rand_master      = 1;               }
        else if (strcmp(argv[i], "-msgsize")      == 0) { msg_size            = atoi(arg_value(argc, argv, &i)); }
        else if (strcmp(argv[i], "-endl")         == 0) { endless             = 1;               }
        else if (strcmp(argv[i], "-iter")         == 0) { max_iters           = atoi(arg_value(argc, argv, &i)); }
        else if (strcmp(argv[i], "-warmup")       == 0) { warm_up_iters       = atoi(arg_value(argc, argv, &i)); }
        else if (strcmp(argv[i], "-blength")      == 0) { burst_length        = parse_duration_arg("-blength", arg_value(argc, argv, &i)); }
        else if (strcmp(argv[i], "-bpause")       == 0) { burst_pause         = parse_duration_arg("-bpause",  arg_value(argc, argv, &i)); }
        else if (strcmp(argv[i], "-bldist")       == 0) { strncpy(burst_dist, arg_value(argc, argv, &i), 15);
                                                          burst_dist[15] = '\0';
                                                          burst_length_rand = 1;                  }
        else if (strcmp(argv[i], "-bpdist")       == 0) { strncpy(pause_dist, arg_value(argc, argv, &i), 15);
                                                          pause_dist[15] = '\0';
                                                          burst_pause_rand  = 1;                  }
        else if (strcmp(argv[i], "-blshape")      == 0) { burst_shape        = atof(arg_value(argc, argv, &i)); }
        else if (strcmp(argv[i], "-bpshape")      == 0) { pause_shape        = atof(arg_value(argc, argv, &i)); }
        else if (strcmp(argv[i], "-seed")         == 0) { rand_seed           = atoi(arg_value(argc, argv, &i)); }
        else if (strcmp(argv[i], "-grty")         == 0) { measure_granularity = atoi(arg_value(argc, argv, &i)); }
        else if (strcmp(argv[i], "-maxsamples")   == 0) { max_samples         = atoi(arg_value(argc, argv, &i)); }
        else if (strcmp(argv[i], "-pretty-print") == 0) { pretty_output       = 1;               }
        else if (strcmp(argv[i], "-debug")        == 0) { debug_mode          = 1;               }
        else if (strcmp(argv[i], "-gpu")          == 0) { use_gpu             = 1;               }
        else if (strcmp(argv[i], "-plot")         == 0) { plot_output         = 1;               }
        else if (strcmp(argv[i], "-plotstat")     == 0) { strncpy(plot_stat, arg_value(argc, argv, &i), 15);
                                                          plot_stat[15] = '\0';                   }
        else if (strcmp(argv[i], "-plotbins")     == 0) { plot_bins           = atoi(arg_value(argc, argv, &i)); }
        else if (strcmp(argv[i], "-plotlog")      == 0) { plot_log            = 1;               }
        else if (strcmp(argv[i], "-plotbinsize")  == 0) { const char *_bs = arg_value(argc, argv, &i);
                                                          plot_bin_size = parse_duration(_bs);
                                                          if (isnan(plot_bin_size) || plot_bin_size <= 0.0) {
                                                              if (my_rank == master_rank)
                                                                  fprintf(stderr, "-plotbinsize must be a positive duration "
                                                                          "(e.g. 2ms, 500us, 0.001), got '%s'\n", _bs);
                                                              MPI_Abort(MPI_COMM_WORLD, -1);
                                                          }                                       }
        else { argv[new_argc++] = argv[i]; } /* pass through unknown flags */
    }

    /* fail fast on nonsensical numeric parameters */
    if (max_samples < 1) {
        if (my_rank == master_rank)
            fprintf(stderr, "-maxsamples must be >= 1, got %d\n", max_samples);
        MPI_Abort(MPI_COMM_WORLD, -1);
    }
    if (measure_granularity < 1) {
        if (my_rank == master_rank)
            fprintf(stderr, "-grty must be >= 1, got %d\n", measure_granularity);
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    /* a randomised burst/pause distribution needs a positive mean and a valid
     * shape — validate now so we never feed log(0)/negative values to a sampler */
    if (burst_length_rand)
        validate_burst_dist("-bldist", burst_dist, burst_length, burst_shape);
    if (burst_pause_rand)
        validate_burst_dist("-bpdist", pause_dist, burst_pause, pause_shape);

    /* seed RNG (shared across all ranks) and resolve randomised master */
    srand(rand_seed);
    if (do_rand_master)
        master_rank = rand() % w_size;

    /* validate the resolved master rank is a real rank.  An out-of-range -mrank
     * would otherwise become an invalid root in MPI_Gather/Bcast/Reduce, and can
     * deadlock benchmarks that branch on (my_rank == master_rank).  Gate the
     * message on rank 0 because master_rank itself may be the bad value.       */
    if (master_rank < 0 || master_rank >= w_size) {
        if (my_rank == 0)
            fprintf(stderr, "-mrank must be in [0, %d), got %d\n", w_size, master_rank);
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    /* validate -plot options */
    if (plot_output) {
        if (strcmp(plot_stat, "avg") && strcmp(plot_stat, "min") && strcmp(plot_stat, "max")
            && strcmp(plot_stat, "median") && strcmp(plot_stat, "mainrank")) {
            if (my_rank == master_rank)
                fprintf(stderr, "-plotstat must be one of avg|min|max|median|mainrank, got %s\n", plot_stat);
            MPI_Abort(MPI_COMM_WORLD, -1);
        }
        if (plot_bin_size > 0.0 && plot_log) {
            if (my_rank == master_rank)
                fprintf(stderr, "-plotbinsize cannot be combined with -plotlog "
                                "(a constant width is meaningless on a log axis)\n");
            MPI_Abort(MPI_COMM_WORLD, -1);
        }
        if (plot_bin_size <= 0.0 && plot_bins < 1) {
            if (my_rank == master_rank)
                fprintf(stderr, "-plotbins must be >= 1, got %d\n", plot_bins);
            MPI_Abort(MPI_COMM_WORLD, -1);
        }
    }

    gpu_setup();
    return new_argc;
}

/* Convenience wrappers used by benchmark measurement loops. */
double sample_burst_length(double mean) { return rand_duration(mean, burst_dist, burst_shape); }
double sample_pause_length(double mean) { return rand_duration(mean, pause_dist, pause_shape); }

/* Record a single measured-iteration latency into the ring buffer.
 * Skips writes during warm-up: callers gate this by `if (k >= warm_up_iters)`
 * at the outer iteration level — see the standard benchmark loop.
 *
 * The ring buffer of size `max_samples` therefore only ever holds measured
 * samples (no warmup pollution); the (oldest → newest) ordering is
 * (measured_iters % max_samples) ... (measured_iters - 1) % max_samples.   */
void record_duration(double t)
{
    durations[measured_iters % max_samples] = t;
    measured_iters++;
}

/* ── output helpers ──────────────────────────────────────────────────────── */

/*format a duration (seconds) with auto-scaled units; "%9.2f UU" is normally
 * 12 chars wide (9 is a minimum field width, so extreme outliers can exceed it)*/
void format_duration(char *buf, size_t len, double t)
{
    if (t < 1e-3)
        snprintf(buf, len, "%9.2f us", t * 1e6);
    else if (t < 1.0)
        snprintf(buf, len, "%9.2f ms", t * 1e3);
    else
        snprintf(buf, len, "%9.2f  s", t);
}

/* Strip the leading spaces format_duration adds (it right-pads to %9.2f). */
const char *ltrim_spaces(const char *s)
{
    while (*s == ' ') s++;
    return s;
}

/* Print one histogram row: an aligned [lo - hi] range label, a bar scaled to
 * the tallest bin (`maxc`), the empirical percentage, and — when `theory` >= 0
 * — a theoretical percentage alongside it (dist_test's CDF overlay).  An
 * infinite `hi` is rendered as "inf" for an open-ended overflow bin.  Shared by
 * the -plot histogram and dist_test's sampler-verification histogram.         */
void print_histogram_row(double lo, double hi, int count, int maxc,
                         int n, double theory)
{
    char lo_s[24], hi_s[24];
    int  k, bar;
    format_duration(lo_s, sizeof(lo_s), lo);
    if (isinf(hi)) snprintf(hi_s, sizeof(hi_s), "    %8s", "inf");
    else           format_duration(hi_s, sizeof(hi_s), hi);
    bar = (maxc > 0) ? (int)floor((double)count / maxc * BLINK_PLOT_BAR_MAX + 0.5) : 0;
    printf("  [%s - %s]  ", lo_s, hi_s);
    for (k = 0; k < bar; k++)                  printf("\xe2\x96\x88"); /* full block */
    for (k = bar; k < BLINK_PLOT_BAR_MAX; k++) printf(" ");
    if (theory >= 0.0) printf("  %5.1f%%  (theory %5.1f%%)\n", 100.0 * count / n, 100.0 * theory);
    else               printf("  %5.1f%%\n", 100.0 * count / n);
}

/* Render an ASCII histogram of `n` per-iteration timing samples (seconds),
 * followed by a percentile footer.  Binning modes:
 *   logscale     : `bins` geometric bins between min and max (needs min > 0);
 *   bin_size > 0 : fixed-width linear bins, lower edge aligned to a multiple of
 *                  bin_size, capped at BLINK_PLOT_MAX_BINS with the tail folded
 *                  into the last bin (a note is printed when this triggers);
 *   otherwise    : `bins` equal-width linear bins between min and max.
 * `vals` is sorted in place — pass a scratch array you no longer need.        */
void print_runtime_histogram(double *vals, int n, const char *statname,
                             int bins, double bin_size, int logscale)
{
    int i;
    printf("\n");
    if (n <= 0) { printf("  (-plot: no measured samples to plot)\n"); return; }

    qsort(vals, n, sizeof(double), compare_doubles);
    double vmin = vals[0], vmax = vals[n - 1];

    double sum = 0.0;
    for (i = 0; i < n; i++) sum += vals[i];

    if (vmax <= vmin) {                       /* degenerate: all identical */
        char one[24];
        format_duration(one, sizeof(one), vmin);
        printf("\033[1m\033[36m  Per-iteration latency  \xc2\xb7  stat=%s  \xc2\xb7  n=%d\033[0m\n",
               statname, n);
        printf("  all samples = %s\n", ltrim_spaces(one));
        return;
    }

    int    use_log  = (logscale && vmin > 0.0);
    int    nb;
    int    overflow = 0;
    double lo0 = vmin, width = 0.0, ratio = 1.0;

    if (use_log) {
        nb = bins > 0 ? bins : 10;
        ratio = pow(vmax / vmin, 1.0 / nb);
    } else if (bin_size > 0.0) {
        lo0 = floor(vmin / bin_size) * bin_size;
        long need = (long)floor((vmax - lo0) / bin_size) + 1;
        if (need < 1) need = 1;
        if (need > BLINK_PLOT_MAX_BINS) { nb = BLINK_PLOT_MAX_BINS; overflow = 1; }
        else                            { nb = (int)need; }
        width = bin_size;
    } else {
        nb = bins > 0 ? bins : 10;
        width = (vmax - vmin) / nb;
    }
    if (nb < 1) nb = 1;

    int *counts = (int *)calloc((size_t)nb, sizeof(int));
    if (!counts) { printf("  (-plot: allocation failed)\n"); return; }
    for (i = 0; i < n; i++) {
        int b = use_log ? (int)floor(log(vals[i] / vmin) / log(ratio))
                        : (int)floor((vals[i] - lo0) / width);
        if (b < 0)   b = 0;
        if (b >= nb) b = nb - 1;              /* fold any overflow into the last bin */
        counts[b]++;
    }
    int maxc = 1;
    for (i = 0; i < nb; i++) if (counts[i] > maxc) maxc = counts[i];

    printf("\033[1m\033[36m  Per-iteration latency  \xc2\xb7  stat=%s  \xc2\xb7  n=%d  \xc2\xb7  %s\033[0m\n",
           statname, n, use_log ? "log bins" : (bin_size > 0.0 ? "fixed bins" : "linear bins"));
    if (overflow)
        printf("\033[33m  note: -plotbinsize would need %ld bins for the observed range; "
               "capped at %d (tail folded into the last bin)\033[0m\n",
               (long)floor((vmax - lo0) / bin_size) + 1, BLINK_PLOT_MAX_BINS);
    printf("\033[2m  ----------------------------------------------------------------------\033[0m\n");

    for (i = 0; i < nb; i++) {
        double lo = use_log ? vmin * pow(ratio, i)     : lo0 + i * width;
        double hi = use_log ? vmin * pow(ratio, i + 1) : lo0 + (i + 1) * width;
        print_histogram_row(lo, hi, counts[i], maxc, n, -1.0);   /* -1 => no theory overlay */
    }
    printf("\033[2m  ----------------------------------------------------------------------\033[0m\n");

    char s_min[24], s_mean[24], s_p50[24], s_p90[24], s_p99[24], s_max[24];
    format_duration(s_min,  sizeof(s_min),  vmin);
    format_duration(s_mean, sizeof(s_mean), sum / n);
    format_duration(s_p50,  sizeof(s_p50),  vals[(int)(0.50 * (n - 1) + 0.5)]);
    format_duration(s_p90,  sizeof(s_p90),  vals[(int)(0.90 * (n - 1) + 0.5)]);
    format_duration(s_p99,  sizeof(s_p99),  vals[(int)(0.99 * (n - 1) + 0.5)]);
    format_duration(s_max,  sizeof(s_max),  vmax);
    printf("  n=%d \xc2\xb7 min %s \xc2\xb7 mean %s \xc2\xb7 p50 %s \xc2\xb7 p90 %s \xc2\xb7 p99 %s \xc2\xb7 max %s\n",
           n, ltrim_spaces(s_min), ltrim_spaces(s_mean), ltrim_spaces(s_p50),
           ltrim_spaces(s_p90), ltrim_spaces(s_p99), ltrim_spaces(s_max));
    free(counts);
}

void write_results(void)
{
    double duration_sum;
    double duration_median;
    int num_samples;
    int i;
    int start_index;
    double *tmp_buf = NULL;

    if (measured_iters > max_samples) /* wrapped the sampling ring buffer */
    {
        num_samples = max_samples;
        start_index = (int)(measured_iters % max_samples);
    }
    else
    {
        num_samples = (int)measured_iters;
        start_index = 0;
    }

    /* MPI_Gather below uses num_samples as BOTH send- and recv-count, which is
     * only valid if every rank contributes the same count.  Ranks normally stay
     * in lockstep, but to be robust (e.g. a SIGUSR1 shutdown that reaches ranks
     * at slightly different iterations) collectively agree on the common minimum
     * so the gather can never mismatch.                                        */
    {
        int common_samples = num_samples;
        MPI_Allreduce(&num_samples, &common_samples, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        /* keep only the most recent common_samples (drop the oldest extras) */
        start_index += (num_samples - common_samples);
        num_samples  = common_samples;
    }

    tmp_buf = (double *)malloc(sizeof(double) * (num_samples > 0 ? num_samples : 1));
    /* copy in chronological order (oldest → newest) using circular indexing */
    for (i = 0; i < num_samples; i++) {
        tmp_buf[i] = durations[(start_index + i) % max_samples];
    }

    double *all_data    = (double *)malloc(sizeof(double)*(num_samples > 0 ? num_samples : 1)*w_size);
    double *sorting_buf = (double *)malloc(sizeof(double)*w_size);

    if (all_data == NULL || sorting_buf == NULL) {
        fprintf(stderr, "Failed to allocate a buffer on rank %d\n", my_rank);
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    /* print header before the gather so it appears first */
    if (my_rank == master_rank) {
        if (pretty_output) {
            printf("\033[1m\033[36m"
                   "  %5s  %12s  %12s  %12s  %12s"
                   "\033[0m\n",
                   "#", "avg", "min", "max", "median");
            printf("\033[2m"
                   "  -------------------------------------------------------------"
                   "\033[0m\n");
        } else {
            printf("Average,Minimum,Maximum,Median,MainRank\n");
        }
    }

    MPI_Gather(tmp_buf, num_samples, MPI_DOUBLE,
               all_data, num_samples, MPI_DOUBLE,
               master_rank, MPI_COMM_WORLD);

    if (my_rank == master_rank) {
        /* -plot: capture one chosen cross-rank statistic per iteration */
        int     plot_stat_id = 0;   /* 0=avg 1=min 2=max 3=median 4=mainrank */
        double *plot_vals    = NULL;
        if (plot_output) {
            if      (strcmp(plot_stat, "min")      == 0) plot_stat_id = 1;
            else if (strcmp(plot_stat, "max")      == 0) plot_stat_id = 2;
            else if (strcmp(plot_stat, "median")   == 0) plot_stat_id = 3;
            else if (strcmp(plot_stat, "mainrank") == 0) plot_stat_id = 4;
            plot_vals = (double *)malloc(sizeof(double) * (num_samples > 0 ? num_samples : 1));
            if (plot_vals == NULL) {
                fprintf(stderr, "Failed to allocate plot buffer on rank %d\n", my_rank);
                MPI_Abort(MPI_COMM_WORLD, -1);
            }
        }
        for (i = 0; i < num_samples; i++) {
            int j;
            duration_sum = 0;
            for (j = 0; j < w_size; j++) {
                sorting_buf[j] = all_data[j*num_samples + i];
                duration_sum  += sorting_buf[j];
            }

            qsort(sorting_buf, w_size, sizeof(double), compare_doubles);
            if (w_size % 2 == 0) /* even: median as mean of two middle values */
                duration_median = (sorting_buf[(w_size-1)/2] + sorting_buf[w_size/2]) / 2.0;
            else                 /* odd: median is the middle value */
                duration_median = sorting_buf[(w_size-1)/2];

            if (plot_output) {
                switch (plot_stat_id) {
                    case 1:  plot_vals[i] = sorting_buf[0];          break;
                    case 2:  plot_vals[i] = sorting_buf[w_size - 1]; break;
                    case 3:  plot_vals[i] = duration_median;         break;
                    case 4:  plot_vals[i] = tmp_buf[i];              break;
                    default: plot_vals[i] = duration_sum / w_size;   break;
                }
            }

            if (pretty_output) {
                char avg_s[24], min_s[24], max_s[24], med_s[24];
                format_duration(avg_s, sizeof(avg_s), duration_sum / w_size);
                format_duration(min_s, sizeof(min_s), sorting_buf[0]);
                format_duration(max_s, sizeof(max_s), sorting_buf[w_size-1]);
                format_duration(med_s, sizeof(med_s), duration_median);
                printf("  %5d  %12s  %12s  %12s  %12s\n",
                       i+1, avg_s, min_s, max_s, med_s);
            } else {
                printf("%.9f,%.9f,%.9f,%.9f,%.9f\n",
                       duration_sum / w_size,
                       sorting_buf[0], sorting_buf[w_size-1],
                       duration_median, tmp_buf[i]);
            }
        }

        if (pretty_output) {
            printf("\033[2m"
                   "  -------------------------------------------------------------"
                   "\033[0m\n");
            printf("  %d samples \xc2\xb7 %lld iterations total\n", num_samples, curr_iters);
        } else {
            printf("Ran %lld iterations. Measured %d iterations.\n", curr_iters, num_samples);
        }

        /* -plot: additive ASCII histogram of the chosen per-iteration statistic */
        if (plot_output) {
            print_runtime_histogram(plot_vals, num_samples, plot_stat,
                                    plot_bins, plot_bin_size, plot_log);
            free(plot_vals);
        }

        fflush(stdout);
    }

    free(sorting_buf);
    free(all_data);
    free(tmp_buf);
}

/* Async-signal-safe handler: only sets a flag.  The measurement loop in each
 * benchmark observes the flag and exits cleanly; write_results() and
 * MPI_Finalize() happen on the main thread, never from signal context.    */
void sig_handler(int sig)
{
    (void)sig;
    shutdown_requested = 1;
}

/* Install the SIGUSR1 shutdown handler with sigaction() rather than signal(),
 * for portable, persistent (non-one-shot) semantics across platforms.       */
void install_shutdown_handler(void)
{
    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = sig_handler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = SA_RESTART;
    sigaction(SIGUSR1, &sa, NULL);
}

/* Collectively decide whether to shut down.  SIGUSR1 only sets the flag on the
 * rank that received it, so a purely local check would make ranks leave the
 * measurement loop at different iterations — desynchronising the per-iteration
 * collectives and the final MPI_Gather (deadlock / mismatch).  This all-reduce
 * (called once per outer iteration, OUTSIDE the timed region) guarantees every
 * rank breaks on the same iteration.  Returns non-zero if any rank requested
 * shutdown.                                                                   */
int check_shutdown(void)
{
    int local = (int)shutdown_requested;
    int global = 0;
    MPI_Allreduce(&local, &global, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    return global;
}

/* ── combinatorics helpers ───────────────────────────────────────────────── */

/*use Fisher-Yates to permute array*/
void permute(int *a, int n)
{
    int j, t, i;
    for (i = n; i > 1; i--)
    {
        j = rand() % i;
        t = a[i - 1];
        a[i - 1] = a[j];
        a[j] = t;
    }
}

/*mathematical mod without negative numbers*/
int mod(int a, int b)
{
    int c = a % b;
    if (c < 0)
        c += b;
    return c;
}

/* Produce a uniformly random matching of [0..n-1] (each entry pairs with
 * exactly one other entry).  Algorithm: shuffle the index list, then pair
 * adjacent elements.  This is provably uniform over the set of perfect
 * matchings (and far simpler than the prior slot-counting version).
 *
 * For odd n the last element targets itself (it has no partner).
 */
void random_pairs(int *a, int n)
{
    int i;
    int has_self = (n % 2 == 1);
    int pair_n   = has_self ? n - 1 : n;
    int *shuf    = (int *)malloc(sizeof(int) * n);

    if (shuf == NULL) {
        fprintf(stderr, "random_pairs: malloc failed\n");
        MPI_Abort(MPI_COMM_WORLD, -1);
    }
    for (i = 0; i < n; i++) shuf[i] = i;
    permute(shuf, n);

    for (i = 0; i < n; i++) a[i] = -1;

    for (i = 0; i < pair_n; i += 2) {
        int x = shuf[i];
        int y = shuf[i + 1];
        a[x] = y;
        a[y] = x;
    }
    if (has_self) {
        /* odd n: one rank has no partner — target itself */
        int lone = shuf[n - 1];
        a[lone] = lone;
    }
    free(shuf);
}

/* Produce fixed-offset pairs: walks i from 0 upward and pairs i with (i+o)
 * whenever both ranks are still free and the partner is not a wrap-around
 * (n - i >= o).  This guarantees disjoint pairs (i, i+o), (i+2o, i+3o), …
 * — for offset o=1 you get (0,1) (2,3) (4,5)…, for o=2 you get (0,2)
 * (1,3) (4,6) (5,7)…, etc.
 *
 * Only non-wrapping reciprocal pairs (i, i+o) are formed; any rank that cannot
 * be matched without wrapping around falls through to a self-pair (a[i] = i,
 * i.e. it performs no communication).  This happens for every offset > n/2, and
 * also for many offsets <= n/2 when n is not a multiple of 2*o (e.g. n=12, o=4
 * leaves ranks 8..11 self-paired).  Full disjoint tiling is only guaranteed when
 * 2*o divides n (notably o=1).  Recommended usage: 1 <= o <= n/2.
 */
void offset_pairs(int *a, int n, int o)
{
    int i, t;
    for (i = 0; i < n; i++)
        a[i] = -1;
    for (i = 0; i < n; i++)
    {
        t = mod(i + o, n);
        if (a[i] == -1 && a[t] == -1)
        {
            if (n - i >= o)
            {
                a[i] = t;
                a[t] = i;
            }
        }
    }
    for (i = 0; i < n; i++)
    {
        if (a[i] == -1)
            a[i] = i; /* no reciprocal partner — self */
    }
}

void* malloc_align(size_t size)
{
    void *p = NULL;
    int ret = posix_memalign(&p, ALIGNMENT, size);
    if (ret != 0) {
        fprintf(stderr, "Failed to allocate memory on rank\n");
        MPI_Abort(MPI_COMM_WORLD, -1);
    }
    return p;
}
