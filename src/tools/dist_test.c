/*
 * dist_test.c — verify that rand_exp, rand_pareto, and rand_lognormal produce
 * samples that match their theoretical distributions.
 *
 * Each case is checked with:
 *   • an ASCII histogram comparing empirical vs theoretical density
 *   • empirical mean  (tolerance: < 1 % error)
 *   • empirical variance where the theoretical value is well-conditioned (< 10 %)
 *   • one-sample KS test at 1 % significance
 *
 * Run as a single-rank job:
 *   mpirun -n 1 build/bin/dist_test
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "common.h"

/* No Data buffers: -gpu is accepted and has no effect. */
int benchmark_has_data_buffers = 0;

#define N_SAMPLES     100000
#define HIST_NBINS    8

/* Bin edges expressed as multiples of mean.  The last bin is open-ended (overflow). */
static const double EDGES[] = { 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0 };

/* ── theoretical CDF functions ──────────────────────────────────────────── */

static double cdf_exp(double x, double mean, double shape)
{
    (void)shape;
    if (x < 0.0) return 0.0;
    return 1.0 - exp(-x / mean);
}

static double cdf_pareto(double x, double mean, double alpha)
{
    double x_m = mean * (alpha - 1.0) / alpha;
    if (x < x_m) return 0.0;
    return 1.0 - pow(x_m / x, alpha);
}

static double cdf_lognormal(double x, double mean, double sigma)
{
    if (x <= 0.0) return 0.0;
    double mu = log(mean) - 0.5 * sigma * sigma;
    double z  = (log(x) - mu) / sigma;
    return 0.5 * erfc(-z / M_SQRT2);
}

typedef double (*cdf_fn)(double x, double p1, double p2);

/* ── KS statistic D_n ───────────────────────────────────────────────────── */

static double ks_statistic(double *sorted, int n, cdf_fn cdf, double p1, double p2)
{
    double D = 0.0;
    int i;
    for (i = 0; i < n; i++) {
        double F  = cdf(sorted[i], p1, p2);
        double d1 = fabs(F - (double)(i + 1) / n);
        double d2 = fabs(F - (double)i / n);
        double d  = d1 > d2 ? d1 : d2;
        if (d > D) D = d;
    }
    return D;
}

static double ks_critical(int n)
{
    return 1.628 / sqrt((double)n); /* 1 % significance, Kolmogorov approx */
}

/* ── ASCII histogram ─────────────────────────────────────────────────────── */

static void print_histogram(double *sorted, int n,
                             cdf_fn cdf, double mean, double p1, double p2)
{
    int counts[HIST_NBINS] = {0};
    int b, i;

    /* bin the (already sorted) samples using linear edges relative to mean */
    b = 0;
    for (i = 0; i < n; i++) {
        while (b < HIST_NBINS - 1 && sorted[i] >= EDGES[b + 1] * mean)
            b++;
        counts[b]++;
    }

    /* find the tallest bar for scaling */
    int max_count = 1;
    for (b = 0; b < HIST_NBINS; b++)
        if (counts[b] > max_count) max_count = counts[b];

    printf("\n");
    for (b = 0; b < HIST_NBINS; b++) {
        double lo     = EDGES[b] * mean;
        double hi     = (b < HIST_NBINS - 1) ? EDGES[b + 1] * mean : INFINITY;
        double theory = cdf(isinf(hi) ? 1e300 : hi, p1, p2) - cdf(lo, p1, p2);
        /* shared renderer (common.h): range label, scaled bar, empirical %,
         * and the theoretical % overlay (theory >= 0) */
        print_histogram_row(lo, hi, counts[b], max_count, n, theory);
    }
    printf("\n");
}

/* ── sampler wrappers (uniform signature) ───────────────────────────────── */

static double wrap_exp      (double mean, double shape) { return rand_exp(mean, shape);      }
static double wrap_pareto   (double mean, double shape) { return rand_pareto(mean, shape);   }
static double wrap_lognormal(double mean, double shape) { return rand_lognormal(mean, shape);}

/* ── run one test case ──────────────────────────────────────────────────── */

static int run_test(const char *label,
                    cdf_fn      cdf,
                    double    (*sampler)(double, double),
                    double      mean,
                    double      shape,
                    double      theoretical_var) /* NAN = skip variance check */
{
    double *samples = malloc(sizeof(double) * N_SAMPLES);
    int i, all_pass = 1;

    for (i = 0; i < N_SAMPLES; i++)
        samples[i] = sampler(mean, shape);

    /* empirical mean */
    double sum = 0.0;
    for (i = 0; i < N_SAMPLES; i++) sum += samples[i];
    double emp_mean = sum / N_SAMPLES;

    /* empirical variance */
    double sum2 = 0.0;
    for (i = 0; i < N_SAMPLES; i++) { double d = samples[i] - emp_mean; sum2 += d * d; }
    double emp_var = sum2 / (N_SAMPLES - 1);

    /* sort once — used by both histogram and KS test */
    qsort(samples, N_SAMPLES, sizeof(double), compare_doubles);

    /* histogram first so it appears right under the case header */
    print_histogram(samples, N_SAMPLES, cdf, mean, mean, shape);

    /* KS test */
    double D      = ks_statistic(samples, N_SAMPLES, cdf, mean, shape);
    double D_crit = ks_critical(N_SAMPLES);

    double mean_err = fabs(emp_mean - mean) / mean * 100.0;
    int    mean_ok  = mean_err < 1.0;
    int    ks_ok    = D < D_crit;
    if (!mean_ok || !ks_ok) all_pass = 0;

    printf("  mean err=%5.2f%% %s  |  KS D=%.4f crit=%.4f %s\n",
           mean_err, mean_ok ? "OK  " : "FAIL", D, D_crit, ks_ok ? "OK" : "FAIL");

    if (!isnan(theoretical_var)) {
        double var_err = fabs(emp_var - theoretical_var) / theoretical_var * 100.0;
        int    var_ok  = var_err < 10.0;
        if (!var_ok) all_pass = 0;
        printf("  variance: empirical=%.4e  theoretical=%.4e  err=%5.1f%% %s\n",
               emp_var, theoretical_var, var_err, var_ok ? "OK" : "FAIL");
    }

    free(samples);
    return all_pass;
}

/* ── main ───────────────────────────────────────────────────────────────── */

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &w_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    parse_common_args(argc, argv);   /* accepts -gpu (no effect: no Data buffers) */

    if (my_rank != 0) { MPI_Finalize(); return 0; }

    srand(42); /* fixed seed for reproducibility */

    int total = 0, passed = 0;
    double mean = 1e-3; /* 1 ms — representative MPI latency */

    printf("=== Blink distribution sampler verification ===\n");
    printf("    n=%d samples  |  KS significance=1%%  |  KS critical=%.4f\n",
           N_SAMPLES, ks_critical(N_SAMPLES));
    printf("    histogram bins: [0, 0.5, 1, 1.5, 2, 3, 5, 10, inf] × mean\n");
    printf("    bar height is relative to the tallest bin\n");

    /* ── exponential ─────────────────────────────────────────────────────── */
    printf("\n--- exp(mean=1ms)  [shape unused] ---\n");
    total++;  passed += run_test("exp", cdf_exp, wrap_exp, mean, 1.0, mean * mean);

    /* ── Pareto ──────────────────────────────────────────────────────────── */
    printf("\n--- pareto(mean=1ms, alpha=1.5)  [infinite variance] ---\n");
    total++;  passed += run_test("pareto alpha=1.5",
                                 cdf_pareto, wrap_pareto, mean, 1.5, NAN);

    printf("\n--- pareto(mean=1ms, alpha=2.5)  [infinite kurtosis → var check skipped] ---\n");
    total++;  passed += run_test("pareto alpha=2.5",
                                 cdf_pareto, wrap_pareto, mean, 2.5, NAN);

    printf("\n--- pareto(mean=1ms, alpha=4.0)  [finite kurtosis] ---\n");
    {
        double alpha = 4.0;
        double x_m   = mean * (alpha - 1.0) / alpha;
        double var   = x_m * x_m * alpha / ((alpha-1.0)*(alpha-1.0)*(alpha-2.0));
        total++;  passed += run_test("pareto alpha=4.0",
                                     cdf_pareto, wrap_pareto, mean, alpha, var);
    }

    /* ── log-normal ──────────────────────────────────────────────────────── */
    printf("\n--- lognormal(mean=1ms, sigma=0.5) ---\n");
    {
        double sigma = 0.5;
        double var   = (exp(sigma*sigma) - 1.0) * mean * mean;
        total++;  passed += run_test("lognormal sigma=0.5",
                                     cdf_lognormal, wrap_lognormal, mean, sigma, var);
    }

    printf("\n--- lognormal(mean=1ms, sigma=1.0) ---\n");
    {
        double sigma = 1.0;
        double var   = (exp(sigma*sigma) - 1.0) * mean * mean;
        total++;  passed += run_test("lognormal sigma=1.0",
                                     cdf_lognormal, wrap_lognormal, mean, sigma, var);
    }

    printf("\n--- lognormal(mean=1ms, sigma=1.5)  [huge kurtosis → var check skipped] ---\n");
    {
        double sigma = 1.5;
        total++;  passed += run_test("lognormal sigma=1.5",
                                     cdf_lognormal, wrap_lognormal, mean, sigma, NAN);
    }

    printf("\n%d / %d tests passed.\n", passed, total);

    MPI_Finalize();
    return (passed == total) ? 0 : 1;
}
