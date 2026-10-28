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
 * a ring MPI_Sendrecv and an MPI_Allreduce(MAX) over ints (MAX: no overflow at any rank count).  Returns 1 on all ranks if correct. */
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
    MPI_Allreduce(ds, dr, N, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    gpu_memcpy_d2h(r, dr, sizeof r);
    for (i = 0; i < N; i++) if (r[i] != w_size) ok = 0;

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
 * While a guard is on, such a crash prints "rank R: <msg>" and exits non-zero.  Every crashing rank prints: the
 * launcher kills the job as soon as one rank dies, so a message left to rank 0 is lost when another rank faults first. */
static char             crash_msg[512];
static size_t           crash_len = 0;   /* 0: no guard on */
static struct sigaction old_segv, old_bus;

static void on_probe_crash(int sig)
{
    (void)sig;
    if (crash_len) {
        ssize_t w = write(STDERR_FILENO, crash_msg, crash_len);   /* write(): async-signal-safe */
        (void)w;
    }
    _exit(1);
}

static void crash_guard_on(const char *msg)
{
    struct sigaction sa;
    int n = snprintf(crash_msg, sizeof crash_msg, "rank %d: %s", my_rank, msg);
    crash_len = (n < 0) ? 0 : ((size_t)n < sizeof crash_msg ? (size_t)n : sizeof crash_msg - 1);
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
    crash_len = 0;
}

/* Nonblocking reductions: some MPIs (Open MPI's libnbc) apply the reduction operator on the CPU and
 * crash on device pointers.  Probe every operation the nonblocking-reduction benchmarks use. */
static int probe_nb_reduce(void)
{
    enum { N = 16 };
    int want = w_size;   /* MAX over my_rank + 1 */
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
    MPI_Iallreduce(ds, dr, N, MPI_INT, MPI_MAX, MPI_COMM_WORLD, &req);
    MPI_Wait(&req, MPI_STATUS_IGNORE);
    gpu_memcpy_d2h(r, dr, sizeof r);
    for (i = 0; i < N; i++) if (r[i] != want) ok = 0;

    /* MPI_Ireduce (result on rank 0) */
    gpu_memset(dr, 0, sizeof r);
    MPI_Ireduce(ds, dr, N, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD, &req);
    MPI_Wait(&req, MPI_STATUS_IGNORE);
    if (my_rank == 0) {
        gpu_memcpy_d2h(r, dr, sizeof r);
        for (i = 0; i < N; i++) if (r[i] != want) ok = 0;
    }

    /* MPI_Ireduce_scatter (one element per rank) */
    gpu_memset(drs_out, 0, sizeof(int));
    MPI_Ireduce_scatter(drs_in, drs_out, recvcounts, MPI_INT, MPI_MAX, MPI_COMM_WORLD, &req);
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
    int lrank, lsize, ndev, shared, any_shared, dev, j;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, my_rank, MPI_INFO_NULL, &local);
    MPI_Comm_rank(local, &lrank);
    MPI_Comm_size(local, &lsize);
    ndev = gpu_device_count();
    {
        /* Decide together, report once per node (its local rank 0), and only then abort: without the barrier the
         * first rank to abort would kill the reporting ranks before they print. */
        int none = (ndev <= 0), any_none = 0;
        MPI_Allreduce(&none, &any_none, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        if (any_none) {
            if (none && lrank == 0)
                fprintf(stderr, "rank %d (first rank of its node): -gpu: no %s device visible\n",
                        my_rank, gpu_backend_name());
            fflush(stderr);
            MPI_Barrier(MPI_COMM_WORLD);
            MPI_Abort(MPI_COMM_WORLD, -1);
        }
    }
    dev = lrank % ndev;
    gpu_set_device(dev);

    /* Do ranks of this node really share a device?  Compare PCI ids: with launcher GPU binding every rank sees
     * one device (ndev == 1) but each has its own, so counting (lsize > ndev) alone would warn falsely. */
    {
        enum { IDLEN = 32 };
        char id[IDLEN] = {0}, *ids = (char *)calloc((size_t)lsize, IDLEN);
        int known = (gpu_device_pci_id(dev, id, IDLEN) == 0), all_known = 0;
        MPI_Allreduce(&known, &all_known, 1, MPI_INT, MPI_MIN, local);
        MPI_Allgather(id, IDLEN, MPI_CHAR, ids, IDLEN, MPI_CHAR, local);
        shared = 0;
        if (all_known) {
            for (j = 0; j < lsize; j++)
                if (j != lrank && strncmp(ids + (size_t)j * IDLEN, id, IDLEN) == 0) shared = 1;
        } else {
            shared = (lsize > ndev);   /* no id to compare: fall back to counting */
        }
        free(ids);
    }
    MPI_Comm_free(&local);
    MPI_Allreduce(&shared, &any_shared, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if (my_rank == 0) {
        fprintf(stderr, "GPU mode: backend %s, %d device(s) visible on node of rank 0; "
                        "MPI reports GPU support: %s; probing GPU-aware MPI...\n",
                gpu_backend_name(), ndev, mpi_reports_gpu_support());
        if (any_shared)
            fprintf(stderr, "warning: ranks share a GPU on some node (device = local_rank %% #visible devices); "
                            "their timings interfere\n");
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

/* For the Tools (null_dummy, dist_test), which ignore their arguments: honour only -gpu, which then warns that it
 * has no effect (they define benchmark_has_data_buffers = 0).  No other argument is parsed, validated or rejected. */
void gpu_tool_args(int argc, char **argv)
{
    int i;
    for (i = 1; i < argc; i++)
        if (strcmp(argv[i], "-gpu") == 0) use_gpu = 1;
    gpu_setup();
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
