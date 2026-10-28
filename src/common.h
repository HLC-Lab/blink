#ifndef BLINK_COMMON_H
#define BLINK_COMMON_H

/* Shared Blink runtime: declarations only.  Definitions live in common.c, which
 * is compiled once into the `blink_common` library and linked into every
 * benchmark.  (Previously this was a header full of `static inline` functions
 * instantiated per translation unit — cleaner as a single compilation unit, and
 * it gives accurate gcov coverage instead of per-TU copies gcov can't merge.)  */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <signal.h>
#include <stdbool.h>
#include <sched.h>

#ifdef __cplusplus
extern "C" {           /* common.c is C; let the C++ (comm_only) benchmarks link to it */
#endif

/* ── globals (defined in common.c) ───────────────────────────────────────── */
extern int           my_rank;
extern int           w_size;
extern int           master_rank;
extern long long     curr_iters;        /* total outer iterations (warmup + measured) */
extern long long     measured_iters;    /* recorded samples (excludes warmup)         */
extern int           warm_up_iters;
extern int           max_samples;
extern double       *durations;
extern volatile sig_atomic_t shutdown_requested;

extern int           msg_size;
extern int           measure_granularity;
extern int           rand_seed;
extern int           max_iters;
extern int           endless;
extern double        burst_length;
extern int           burst_length_rand;   /* set by -bldist */
extern double        burst_pause;
extern int           burst_pause_rand;    /* set by -bpdist */
extern int           pretty_output;
extern int           debug_mode;

extern int           plot_output;         /* -plot                                  */
extern char          plot_stat[16];       /* avg | min | max | median | mainrank    */
extern int           plot_bins;
extern double        plot_bin_size;       /* 0 => use plot_bins                     */
extern int           plot_log;

extern char          burst_dist[16];      /* "exp" | "pareto" | "lognormal"         */
extern double        burst_shape;
extern char          pause_dist[16];
extern double        pause_shape;

/* Benchmark-specific help text, printed after the common block by -h/--help.
 * Weakly defined as NULL in common.c; benchmarks that take their own flags
 * provide a strong definition at file scope (see e.g. src/pairwise/pairwise_b.c).
 */
extern const char   *benchmark_help;

/* ── functions (defined in common.c) ─────────────────────────────────────── */
double      rand_exp(double mean, double shape);
double      rand_pareto(double mean, double shape);
double      rand_lognormal(double mean, double sigma);
double      rand_duration(double mean, const char *dist, double shape);
int         dsleep(double t);
int         compare_doubles(const void *p1, const void *p2);
const char *arg_value(int argc, char **argv, int *i);
void        validate_burst_dist(const char *what, const char *dist, double mean, double shape);
double      parse_duration(const char *s);
void        print_help(const char *progname);
int         parse_common_args(int argc, char **argv);
double      sample_burst_length(double mean);
double      sample_pause_length(double mean);
void        record_duration(double t);
void        format_duration(char *buf, size_t len, double t);
const char *ltrim_spaces(const char *s);
void        print_histogram_row(double lo, double hi, int count, int maxc, int n, double theory);
void        print_runtime_histogram(double *vals, int n, const char *statname,
                                     int bins, double bin_size, int logscale);
void        write_results(void);
void        sig_handler(int sig);
void        install_shutdown_handler(void);
int         check_shutdown(void);
void        permute(int *a, int n);
int         mod(int a, int b);
void        random_pairs(int *a, int n);
void        offset_pairs(int *a, int n, int o);
void       *malloc_align(size_t size);

/* ── GPU mode (src/gpu_mode.c) ───────────────────────────────────────────── */
extern int use_gpu;                       /* set by -gpu; cleared again for Buffer-less executables */
extern int benchmark_has_data_buffers;    /* weak default 1; Buffer-less executables define 0        */
extern int benchmark_uses_rma;            /* weak default 0; incast_get / incast_put define 1        */
extern int benchmark_uses_nb_reduce;      /* weak default 0; the MPI_I{allreduce,reduce,reduce_scatter} benchmarks define 1 */
void  gpu_setup(void);                    /* called at the end of parse_common_args                 */
void  gpu_tool_args(int argc, char **argv); /* Tools that ignore their arguments: honour -gpu only   */
void *data_mirror(void *host, size_t n);                    /* host ptr in Host mode, else device copy of host[0..n) */
void  data_push(void *mirror, const void *host, size_t n);  /* host -> mirror (no-op in Host mode)   */
void  data_fetch(void *host, const void *mirror, size_t n); /* mirror -> host (no-op in Host mode)   */
void  data_free(void *mirror);                              /* no-op in Host mode                    */
void  data_copy(void *dst, const void *src, size_t n);      /* memcpy on mirrors (device-to-device in GPU mode) */

#ifdef __cplusplus
}
#endif

#endif /* BLINK_COMMON_H */
