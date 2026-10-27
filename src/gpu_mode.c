#include "common.h"
#include "gpu.h"

#if defined(OPEN_MPI) && OPEN_MPI
#  include <mpi-ext.h>   /* MPIX_CUDA_AWARE_SUPPORT / MPIX_ROCM_AWARE_SUPPORT */
#endif

/* "yes" | "no" | "unknown ..." according to the MPI's own query, when it has one. */
static const char *mpi_reports_gpu_support(void)
{
#if defined(MPIX_CUDA_AWARE_SUPPORT)
    if (strcmp(gpu_backend_name(), "CUDA") == 0) {
#  if MPIX_CUDA_AWARE_SUPPORT
        return MPIX_Query_cuda_support() ? "yes" : "no";
#  else
        return "no (MPI built without CUDA support)";
#  endif
    }
#endif
#if defined(MPIX_ROCM_AWARE_SUPPORT)
    if (strcmp(gpu_backend_name(), "HIP") == 0) {
#  if MPIX_ROCM_AWARE_SUPPORT
        return MPIX_Query_rocm_support() ? "yes" : "no";
#  else
        return "no (MPI built without ROCm support)";
#  endif
    }
#endif
    return "unknown (not queryable with this MPI)";
}

/* Untimed collective check that the MPI moves data from/to device memory:
 * a ring MPI_Sendrecv and an MPI_Allreduce(SUM) over ints.  Returns 1 on all ranks if correct. */
static int probe_pt2pt(void)
{
    enum { N = 16 };
    int h[N], r[N], i, ok = 1, all = 0;
    int *ds = (int *)gpu_malloc(N * sizeof(int));
    int *dr = (int *)gpu_malloc(N * sizeof(int));
    int to = (my_rank + 1) % w_size, from = (my_rank + w_size - 1) % w_size;

    for (i = 0; i < N; i++) h[i] = my_rank + 1;
    gpu_memcpy_h2d(ds, h, sizeof h);
    gpu_memset(dr, 0, sizeof h);
    MPI_Sendrecv(ds, N, MPI_INT, to, 7, dr, N, MPI_INT, from, 7, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    gpu_memcpy_d2h(r, dr, sizeof r);
    for (i = 0; i < N; i++) if (r[i] != from + 1) ok = 0;

    gpu_memset(dr, 0, sizeof h);
    MPI_Allreduce(ds, dr, N, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    gpu_memcpy_d2h(r, dr, sizeof r);
    for (i = 0; i < N; i++) if (r[i] != w_size * (w_size + 1) / 2) ok = 0;

    gpu_free(ds); gpu_free(dr);
    MPI_Allreduce(&ok, &all, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    return all;
}

/* One-sided probe (only for benchmarks that use MPI_Win): Put then Get on a window over
 * device memory.  Some MPIs accept this and silently transfer nothing. */
static int probe_rma(void)
{
    enum { N = 16 };
    int h[N], r[N], i, ok = 1, all = 0;
    int *ds = (int *)gpu_malloc(N * sizeof(int));
    int *dw = (int *)gpu_malloc(N * sizeof(int));
    int *dg = (int *)gpu_malloc(N * sizeof(int));
    int to = (my_rank + 1) % w_size, from = (my_rank + w_size - 1) % w_size;
    MPI_Win win;

    for (i = 0; i < N; i++) h[i] = my_rank + 1;
    gpu_memcpy_h2d(ds, h, sizeof h);
    gpu_memset(dw, 0, sizeof h);
    gpu_memset(dg, 0, sizeof h);
    MPI_Win_create(dw, N * sizeof(int), sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &win);
    MPI_Win_fence(0, win);
    MPI_Put(ds, N, MPI_INT, to, 0, N, MPI_INT, win);
    MPI_Win_fence(0, win);
    gpu_memcpy_d2h(r, dw, sizeof r);
    for (i = 0; i < N; i++) if (r[i] != from + 1) ok = 0;
    MPI_Win_fence(0, win);
    MPI_Get(dg, N, MPI_INT, to, 0, N, MPI_INT, win);   /* window of `to` holds to's predecessor = my_rank + 1 */
    MPI_Win_fence(0, win);
    gpu_memcpy_d2h(r, dg, sizeof r);
    for (i = 0; i < N; i++) if (r[i] != my_rank + 1) ok = 0;
    MPI_Win_free(&win);

    gpu_free(ds); gpu_free(dw); gpu_free(dg);
    MPI_Allreduce(&ok, &all, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    return all;
}

/* Number of device mirrors this rank created.  Under -debug it is printed at exit, so tests can
 * tell a benchmark that really hands device memory to MPI from one that ignores -gpu. */
static int mirrors_created = 0;

static void print_mirror_count(void)
{
    printf("DEBUG rank=%d mirrors=%d\n", my_rank, mirrors_created);
    fflush(stdout);
}

/* Crash guard.  A probe that hands device pointers to an MPI which cannot use them usually dies with
 * SIGSEGV/SIGBUS inside the MPI (the CPU dereferences a device address) instead of returning an error.
 * While a guard is on, such a crash prints `crash_msg` (rank 0 only) and exits non-zero. */
static const char     *crash_msg = NULL;
static struct sigaction old_segv, old_bus;

static void on_probe_crash(int sig)
{
    (void)sig;
    if (my_rank == 0 && crash_msg) {
        ssize_t w = write(STDERR_FILENO, crash_msg, strlen(crash_msg));
        (void)w;
    }
    _exit(1);
}

static void crash_guard_on(const char *msg)
{
    struct sigaction sa;
    crash_msg = msg;
    sa.sa_handler = on_probe_crash;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;
    sigaction(SIGSEGV, &sa, &old_segv);
    sigaction(SIGBUS,  &sa, &old_bus);
}

static void crash_guard_off(void)
{
    sigaction(SIGSEGV, &old_segv, NULL);
    sigaction(SIGBUS,  &old_bus,  NULL);
    crash_msg = NULL;
}

/* Nonblocking reductions: some MPIs (Open MPI's libnbc) apply the reduction operator on the CPU and
 * crash on device pointers.  Probe every operation the nonblocking-reduction benchmarks use. */
static int probe_nb_reduce(void)
{
    enum { N = 16 };
    long long want = (long long)w_size * (w_size + 1) / 2;
    int i, ok = 1, all = 0, *h, *recvcounts;
    int *ds = (int *)gpu_malloc(N * sizeof(int));
    int *dr = (int *)gpu_malloc(N * sizeof(int));
    int *drs_in  = (int *)gpu_malloc((size_t)w_size * sizeof(int));
    int *drs_out = (int *)gpu_malloc(sizeof(int));
    int r[N], one;
    MPI_Request req;

    h = (int *)malloc((size_t)w_size * sizeof(int));
    recvcounts = (int *)malloc((size_t)w_size * sizeof(int));
    for (i = 0; i < N; i++) r[i] = my_rank + 1;
    gpu_memcpy_h2d(ds, r, sizeof r);
    for (i = 0; i < w_size; i++) { h[i] = my_rank + 1; recvcounts[i] = 1; }
    gpu_memcpy_h2d(drs_in, h, (size_t)w_size * sizeof(int));

    /* MPI_Iallreduce */
    gpu_memset(dr, 0, sizeof r);
    MPI_Iallreduce(ds, dr, N, MPI_INT, MPI_SUM, MPI_COMM_WORLD, &req);
    MPI_Wait(&req, MPI_STATUS_IGNORE);
    gpu_memcpy_d2h(r, dr, sizeof r);
    for (i = 0; i < N; i++) if (r[i] != want) ok = 0;

    /* MPI_Ireduce (result on rank 0) */
    gpu_memset(dr, 0, sizeof r);
    MPI_Ireduce(ds, dr, N, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD, &req);
    MPI_Wait(&req, MPI_STATUS_IGNORE);
    if (my_rank == 0) {
        gpu_memcpy_d2h(r, dr, sizeof r);
        for (i = 0; i < N; i++) if (r[i] != want) ok = 0;
    }

    /* MPI_Ireduce_scatter (one element per rank) */
    gpu_memset(drs_out, 0, sizeof(int));
    MPI_Ireduce_scatter(drs_in, drs_out, recvcounts, MPI_INT, MPI_SUM, MPI_COMM_WORLD, &req);
    MPI_Wait(&req, MPI_STATUS_IGNORE);
    gpu_memcpy_d2h(&one, drs_out, sizeof one);
    if (one != want) ok = 0;

    free(h); free(recvcounts);
    gpu_free(ds); gpu_free(dr); gpu_free(drs_in); gpu_free(drs_out);
    MPI_Allreduce(&ok, &all, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    return all;
}

void gpu_setup(void)
{
    if (!use_gpu) return;

    if (!benchmark_has_data_buffers) {
        if (my_rank == 0)
            fprintf(stderr, "warning: -gpu has no effect: this executable has no data buffers\n");
        use_gpu = 0;
        return;
    }
    if (!gpu_available()) {
        if (my_rank == 0)
            fprintf(stderr, "-gpu: built without GPU support; reconfigure with -DBLINK_GPU=CUDA or -DBLINK_GPU=HIP\n");
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    /* node-local rank -> device */
    MPI_Comm local;
    int lrank, lsize, ndev, over, any_over, dev;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, my_rank, MPI_INFO_NULL, &local);
    MPI_Comm_rank(local, &lrank);
    MPI_Comm_size(local, &lsize);
    MPI_Comm_free(&local);
    ndev = gpu_device_count();
    if (ndev <= 0) {
        fprintf(stderr, "rank %d: -gpu: no %s device visible\n", my_rank, gpu_backend_name());
        MPI_Abort(MPI_COMM_WORLD, -1);
    }
    dev = lrank % ndev;
    gpu_set_device(dev);

    over = (lsize > ndev);
    MPI_Allreduce(&over, &any_over, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if (my_rank == 0) {
        fprintf(stderr, "GPU mode: backend %s, %d device(s) visible on node of rank 0; "
                        "MPI reports GPU support: %s; probing GPU-aware MPI...\n",
                gpu_backend_name(), ndev, mpi_reports_gpu_support());
        if (any_over)
            fprintf(stderr, "warning: more ranks than GPUs on some node; ranks share devices (device = local_rank %% #devices)\n");
    }

    crash_guard_on("this MPI is not GPU-aware: it crashed while moving device memory (Sendrecv/Allreduce); "
                   "use a GPU-aware MPI (Open MPI with CUDA/ROCm + UCX, or MPICH_GPU_SUPPORT_ENABLED=1)\n");
    int pt2pt_ok = probe_pt2pt();
    crash_guard_off();
    if (!pt2pt_ok) {
        if (my_rank == 0)
            fprintf(stderr, "GPU-aware MPI probe FAILED: device-memory Sendrecv/Allreduce returned wrong data\n");
        MPI_Abort(MPI_COMM_WORLD, -1);
    }
    if (benchmark_uses_rma) {
        if (my_rank == 0)
            fprintf(stderr, "probing one-sided communication on device memory...\n");
        crash_guard_on("one-sided (MPI_Win/Put/Get) on GPU memory is not supported by this MPI: it crashed\n");
        int rma_ok = probe_rma();
        crash_guard_off();
        if (!rma_ok) {
            if (my_rank == 0)
                fprintf(stderr, "one-sided (MPI_Win/Put/Get) on GPU memory is not supported by this MPI: data was not transferred\n");
            MPI_Abort(MPI_COMM_WORLD, -1);
        }
    }

    if (benchmark_uses_nb_reduce) {
        if (my_rank == 0)
            fprintf(stderr, "probing nonblocking reductions on device memory...\n");
        crash_guard_on("nonblocking reductions (MPI_Iallreduce/Ireduce/Ireduce_scatter) on GPU memory are "
                       "not supported by this MPI: it crashed while reducing device buffers\n");
        int nbr_ok = probe_nb_reduce();
        crash_guard_off();
        if (!nbr_ok) {
            if (my_rank == 0)
                fprintf(stderr, "nonblocking reductions on GPU memory are not supported by this MPI: wrong result\n");
            MPI_Abort(MPI_COMM_WORLD, -1);
        }
    }

    if (debug_mode) {
        printf("DEBUG rank=%d gpu=1 device=%d\n", my_rank, dev);
        fflush(stdout);
        atexit(print_mirror_count);
    }
}

void *data_mirror(void *host, size_t n)
{
    if (!use_gpu) return host;
    void *dev = gpu_malloc(n);
    gpu_memcpy_h2d(dev, host, n);
    mirrors_created++;
    return dev;
}
void data_push(void *mirror, const void *host, size_t n)  { if (use_gpu) gpu_memcpy_h2d(mirror, host, n); }
void data_fetch(void *host, const void *mirror, size_t n) { if (use_gpu) gpu_memcpy_d2h(host, mirror, n); }
void data_free(void *mirror)                              { if (use_gpu) gpu_free(mirror); }
void data_copy(void *dst, const void *src, size_t n)
{
    if (use_gpu) gpu_memcpy_d2d(dst, src, n);
    else         memcpy(dst, src, n);
}
