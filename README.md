<div align="center">
  <img src="assets/blink.svg" width="600" alt="Blink logo" />
</div>

<div align="left">

[![CI](https://github.com/hlc-lab/blink/actions/workflows/ci.yml/badge.svg?branch=dev)](https://github.com/hlc-lab/blink/actions/workflows/ci.yml?query=branch%3Adev)
[![CodeQL](https://github.com/hlc-lab/blink/actions/workflows/codeql.yml/badge.svg?branch=dev)](https://github.com/hlc-lab/blink/actions/workflows/codeql.yml?query=branch%3Adev)
[![codecov](https://codecov.io/gh/hlc-lab/blink/branch/dev/graph/badge.svg)](https://codecov.io/gh/hlc-lab/blink/tree/dev)
[![License: MIT](https://img.shields.io/github/license/hlc-lab/blink)](LICENSE)

</div>


Blink is a collection of MPI benchmarks designed for long-running, in-situ measurement of network behaviour under realistic traffic conditions.  Each benchmark runs for a configurable number of iterations (or endlessly until interrupted), records per-iteration latency on every rank, and emits a CSV summary when it finishes — either naturally or via `SIGUSR1`.

## How Blink differs from other benchmarking suites

Most MPI benchmark suites run a fixed iteration sweep and report a single aggregate table (min / median / max per message size) before exiting. Blink takes a different approach on three axes:

**Temporal evolution.** Blink records one measurement per iteration rather than collapsing everything into a final summary. This lets you observe how latency evolves over time, detect jitter events, and correlate performance spikes with external activity on the system.

**Burst traffic modeling.** Real applications rarely issue communication at a steady, uniform rate. Blink supports configurable burst-pause cycles with pluggable inter-arrival distributions — exponential, Pareto, and log-normal — so the generated traffic pattern can more closely reflect the bursty, heavy-tailed nature of production workloads.

**Long-running operation.** Blink is designed to run alongside real workloads or as a background monitor. It can run endlessly (`-endl`), keeps only the most recent *N* samples in a ring buffer (`-maxsamples`), and responds to `SIGUSR1` with a clean shutdown — collecting and printing whatever has been measured so far before calling `MPI_Finalize`.

## Building

Requirements: CMake ≥ 3.16, an MPI implementation (e.g. OpenMPI, MPICH, Intel MPI).

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

Binaries are placed in `build/bin/`.  Adding a new `.c` or `.cpp` file anywhere under `src/` is enough — CMake picks it up automatically on the next configure.

### GPU support

`BLINK_GPU` selects the GPU backend used by [`-gpu`](#gpu-mode): `AUTO` (default: CUDA if a toolkit is found, else HIP, else a host-only build), `OFF`, `CUDA` or `HIP`.  An explicit `CUDA`/`HIP` fails the configure step when the toolkit is missing; the chosen backend is printed (`BLINK_GPU: requested AUTO, using CUDA`).  No kernels are compiled, so only the CUDA/HIP *runtime library* is needed — no `nvcc` or `hipcc`.  Use one build directory per backend; `CMakePresets.json` (CMake ≥ 3.21) provides them:

```bash
cmake --preset cpu  && cmake --build --preset cpu    # host only          -> build-cpu/
cmake --preset cuda && cmake --build --preset cuda   # NVIDIA             -> build-cuda/
cmake --preset hip  && cmake --build --preset hip    # AMD (ROCm in /opt/rocm) -> build-hip/
```

On a cluster combine this with `-DBLINK_MPI_WRAPPER=ON` after loading the CUDA/ROCm and MPI modules.  A GPU build still runs in host mode unless `-gpu` is passed.


## Testing

Correctness tests are built automatically alongside the benchmarks (requires GTest >= 1.14, fetched automatically if not found on the system).  Pass `-DBLINK_TESTS=OFF` to skip them.

Each test is a single-process GTest binary that spawns the actual benchmark under `mpirun -n N <binary> -debug` via `popen()` and parses structured `DEBUG key=val` output.  Most suites require **8 MPI ranks**; `test_pingpong` also covers a 2-rank case for `pingpong_b`.

### Run all tests

```bash
ctest --test-dir build --output-on-failure
```

| Useful flags | Effect |
|---|---|
| `-j <N>` | Run up to N test suites in parallel |
| `-V` | Always print test output (verbose) |
| `--output-on-failure` | Print output only on failure |
| `-R <regex>` | Run only suites whose name matches |

### Run a single test suite

```bash
ctest --test-dir build -R test_collectives --output-on-failure
```

### Test suites

| Suite | Benchmarks covered | Properties verified |
|---|---|---|
| `test_collectives` | `allgather_b/nb/comm_only`, `allreduce_b/nb`, `alltoall_b/nb/man/comm_only`, `barrier_b/nb`, `broadcast_b/nb`, `gather_b/nb`, `scatter_b/nb`, `reduce_b/nb`, `reduce_scatter_b/nb` | Coverage, communicator size, root rank, data integrity (plain + burst), b/nb agreement, allgather transfer-volume agreement |
| `test_pairwise` | `pairwise_b/nb/bsnbr` | Coverage, validity, symmetry (offpair/rpair), bijection, exact pairs, non-reciprocal modes (perm/rot) (plain + burst) |
| `test_ring` | `ring_nb`, `ring_bsnbr` | Coverage, validity, exact sequential neighbours, consistency (sequential + random), completeness (plain + burst) |
| `test_incast` | `incast_b/nb/bsnbr/get/put` | Coverage, roles (one receiver, N-1 senders), target correctness (plain + burst) |
| `test_pingpong` | `pingpong_b` (2 ranks), `pingpong_pairwise_b` (8 ranks) | Coverage, communicator size, data integrity (plain + burst) |
| `test_kpartners` | `kpartners_nb` | Coverage, communicator size, k value, slot coherence (plain + burst) |
| `test_stencil` | `stencil_2d_nb` | Coverage, communicator size, data integrity, neighbour-graph consistency (plain, periodic, burst) |
| `test_misc` | cross-cutting (`barrier_nb`, `incast_b`, `dist_test`, `alltoall_nb`) | Ring-buffer wrap, warm-up exclusion, pretty-print formatter, `-mrand` determinism, sampler self-check, SIGUSR1 clean shutdown |

### GPU tests

GPU tests need a GPU and a GPU-aware MPI, so they are registered only when `BLINK_GPU_TESTS=ON` — the default when `BLINK_GPU=CUDA|HIP` is requested explicitly (e.g. the presets), off for `AUTO` so a machine that merely has a toolkit keeps a green plain `ctest`.  With them on, every suite also gets a `<suite>_gpu` ctest entry: the same binary run with `BLINK_TEST_GPU=1`, which appends `-gpu` to every benchmark launch and additionally fails a run whose ranks report `gpu=1` but created no device copy.  `test_gpu` covers the `-gpu` runtime itself and `test_gpu_layer` the CUDA/HIP wrapper.  All GPU tests share one `ctest` resource lock so they run one at a time.  The one-sided incast cases and the nonblocking-reduction cases are skipped when the MPI cannot handle device memory for them.

```bash
ctest --test-dir build-cuda -R _gpu --output-on-failure
```

### Overriding MPI launcher

The test binaries bake in the MPI launcher paths at configure time.  Override them at runtime with environment variables:

```bash
BLINK_MPIEXEC=/opt/mpi/bin/mpirun BLINK_NPROC_FLAG=-np BLINK_BIN_DIR=/custom/bin ctest --test-dir build
```

### Continuous integration

Every push and pull request triggers the [CI workflow](.github/workflows/ci.yml): it
builds every benchmark and the test suite on Ubuntu with OpenMPI, runs `ctest`, and
produces a coverage report.  The badges at the top of this file show the latest result.

The build is instrumented via `-DBLINK_COVERAGE=ON` (`-O0 --coverage`, atomic counters
so concurrent MPI ranks don't corrupt the `.gcda` files); a [`gcovr`](https://gcovr.com)
summary of `src/` is printed to the run's job summary, attached as an artifact, and
uploaded to Codecov.  To reproduce a coverage run locally:

```bash
cmake -B build-cov -DBLINK_TESTS=ON -DBLINK_COVERAGE=ON
cmake --build build-cov -j$(nproc)
ctest --test-dir build-cov --output-on-failure
gcovr --root . --filter 'src/' --exclude-unreachable-branches --exclude-throw-branches \
      --exclude-lines-by-pattern '.*MPI_Abort.*' --print-summary
```

## Common flags

Every benchmark understands the following flags:

| Flag | Default | Description |
|------|---------|-------------|
| `-msgsize <B>` | `1024` | Message size in bytes |
| `-iter <N>` | `1` | Number of measured iterations (after warm-up) |
| `-warmup <N>` | `5` | Number of warm-up iterations (not recorded) |
| `-endl` | off | Run endlessly until `SIGUSR1` |
| `-grty <N>` | `1` | Measurement granularity — number of MPI calls batched per timed window |
| `-maxsamples <N>` | `1000` | Maximum recorded samples (ring buffer) |
| `-mrank <R>` | `0` | Rank that collects and prints results |
| `-mrand` | off | Pick master rank randomly |
| `-seed <S>` | `1` | RNG seed (shared across ranks) |
| `-blength <duration>` | `0` | Mean burst length (0 = single shot per iteration). Bare number = seconds; suffixes `s`, `ms`, `us`, `ns` accepted (e.g. `400us`). |
| `-bldist <D>` | — | Randomise burst length using distribution `D` (`exp`, `pareto`, `lognormal`) |
| `-blshape <S>` | `1.5` | Shape parameter for burst distribution (α for Pareto, σ for log-normal) |
| `-bpause <duration>`  | `0` | Mean pause between bursts. Same unit conventions as `-blength`. |
| `-bpdist <D>` | — | Randomise pause length using distribution `D` (`exp`, `pareto`, `lognormal`) |
| `-bpshape <S>` | `1.5` | Shape parameter for pause distribution (α for Pareto, σ for log-normal) |
| `-pretty-print` | off | Human-readable table output instead of CSV (see [Output format](#output-format)) |
| `-plot` | off | Append an ASCII histogram of the per-iteration timing distribution (see [Output format](#output-format)) |
| `-plotstat <S>` | `max` | Which cross-rank statistic to histogram each iteration: `avg`, `min`, `max`, `median`, `mainrank` |
| `-plotbins <N>` | `10` | Number of histogram bins |
| `-plotbinsize <D>` | — | Fixed bin width with optional unit (e.g. `2ms`, `500us`, `0.001`); overrides `-plotbins`, linear bins only |
| `-plotlog` | off | Use logarithmic (geometric) bins — good for heavy-tailed latencies |
| `-gpu` | off | Keep the Data buffers in GPU memory and communicate with them directly (see [GPU mode](#gpu-mode)) |
| `-h`, `--help`, `-help` | — | Print usage (common flags + benchmark-specific options if any) and exit |

> **Tip:** every benchmark accepts `-h` (or `--help`); benchmarks with extra flags (`pairwise_*`, `kpartners_nb`, `ring_*`, `stencil_2d_nb`) append a `Benchmark-specific options` section listing them.

### Burst distributions

When `-bldist` or `-bpdist` is supplied, burst lengths / pauses are drawn independently on each iteration from the chosen distribution, parameterised so that the empirical mean equals `-blength` / `-bpause` regardless of the shape parameter:

| Distribution | Flag value | Shape parameter | Notes |
|---|---|---|---|
| Exponential | `exp` | — (ignored) | Light-tailed; fully determined by its mean |
| Pareto | `pareto` | α (`-blshape` / `-bpshape`, must be > 1) | Heavy-tailed; α ≤ 2 → infinite variance |
| Log-normal | `lognormal` | σ (`-blshape` / `-bpshape`, > 0) | Log-symmetric; larger σ → heavier tail |

```bash
# Pareto bursts (α=2, heavy tail) with exponential pauses
mpirun -n 8 build/bin/alltoall_nb -iter 500 \
    -blength 10ms -bldist pareto  -blshape 2.0 \
    -bpause  50ms -bpdist exp
```

The sampler implementations can be verified independently:

```bash
mpirun -n 1 build/bin/dist_test
```

## GPU mode

`-gpu` runs the same MPI calls on **device pointers** (GPU-aware MPI): each Benchmark keeps its host buffers and host fill/verify code, and hands MPI device copies of them.  Only the Data buffers move to the GPU; the per-iteration timings, request arrays and the result gather stay on the host, and the timed region is unchanged (the device copies are made before it, and `-debug` verification copies back after it).

* **Requirements.**  A binary built with a GPU backend (see [GPU support](#gpu-support)) and an MPI that accepts GPU pointers: Open MPI built with CUDA/ROCm support and UCX (`ompi_info --parsable | grep mpi_built_with_cuda_support`), or Cray MPICH with `MPICH_GPU_SUPPORT_ENABLED=1`.  A binary built without a backend exits with `built without GPU support`.
* **Device selection.**  Rank *r* uses device `local_rank % visible_devices`, set after `MPI_Init`.  With launcher GPU binding (e.g. Slurm) each rank usually sees a single device.  If ranks end up on the same physical GPU (compared by PCI id, so launcher binding of one GPU per rank does not trigger it) a warning is printed: their timings interfere.
* **Startup probe.**  Before the benchmark runs, every `-gpu` run performs an untimed ring `MPI_Sendrecv` and an `MPI_Allreduce` on device buffers and aborts with `GPU-aware MPI probe FAILED` if the data is wrong, or with `this MPI is not GPU-aware` if the MPI crashes dereferencing the device pointer (what a non-GPU-aware MPI does).  The first line also prints what the MPI itself reports about GPU support, when it can be queried.
* **One-sided benchmarks** (`incast_get`, `incast_put`) additionally probe `MPI_Put`/`MPI_Get` on a window over device memory.  Some MPIs accept this and silently transfer nothing; Blink then aborts with `one-sided ... on GPU memory is not supported by this MPI` instead of reporting meaningless timings.  (Checked: works with Open MPI 4.1.6 + UCX 1.16 on Leonardo; not with Open MPI 5.0.11 + UCX on a single-GPU laptop.)
* **Nonblocking reductions** (`allreduce_nb`, `reduce_nb`, `reduce_scatter_nb`) are probed the same way: some MPIs (Open MPI's libnbc) apply the reduction operator on the CPU and crash on device pointers while the blocking reductions work.  Blink runs tiny `MPI_Iallreduce`, `MPI_Ireduce` and `MPI_Ireduce_scatter` calls on device memory first and aborts with `nonblocking reductions ... are not supported by this MPI` instead of a bare segfault.  (Checked on Open MPI 4.1.6 on Leonardo and 5.0.11 on a laptop: both reject them, so use the blocking variants in GPU mode.)
* **Executables without Data buffers** (`barrier_b`, `barrier_nb`, `null_dummy`, `dist_test`) accept `-gpu`, run exactly as before and print a warning that the flag has no effect.  `null_dummy` and `dist_test` still ignore every other argument.
* **Output.**  All GPU messages go to stderr, once, from rank 0; stdout is unchanged.  The one exception is a crash caught by a startup probe: every rank that crashes prints `rank R: <reason>`, because the launcher may kill rank 0 before it can.  Under `-debug` each rank also prints `DEBUG rank=R gpu=1 device=D` and, at exit, `DEBUG rank=R mirrors=N` (the number of device copies it created).
* **GPU memory.**  Each rank allocates one device copy of each of its Data buffers, the same size as the host buffer (e.g. `-msgsize` × number of ranks for the all-to-all style benchmarks, × `-grty` for the nonblocking variants, and the incast receive buffer on every rank for `incast_get`/`incast_put`).  Ranks sharing a GPU add up.  A failed device allocation aborts with the CUDA/HIP error text.

```bash
mpirun -n 4 build-cuda/bin/alltoall_b -gpu -msgsize 1048576 -iter 20
mpirun -n 4 build-cuda/bin/pairwise_nb -gpu -debug      # check=OK per rank, device buffers
```

## Auxiliary tools

Besides the benchmarks, `src/tools/` builds a few helper binaries:

| Binary | Purpose |
|--------|---------|
| `dist_test` | Verifies the burst-distribution samplers against their theoretical CDFs (ASCII histogram + mean / variance checks + KS test).  Run single-rank: `mpirun -n 1 build/bin/dist_test`. |
| `checker` | An all-to-all workload (like `alltoall_b`) that additionally writes a per-iteration wall-clock timestamp log on the master rank (`checker_<YYYYMMDD_HHMMSS>.log`, local time) — useful for correlating throughput dips with external system activity over a long run.  Accepts the common flags. |
| `null_dummy` | Bare `MPI_Init` / `MPI_Finalize` with no communication; a baseline for measuring MPI startup / teardown overhead. |

## Benchmarks

Benchmarks are grouped by traffic pattern.  The suffix convention is:

| Suffix | Meaning |
|--------|---------|
| `_b` | Blocking MPI call |
| `_nb` | Non-blocking MPI call (`MPI_I*` + `MPI_Waitall`) |
| `_bsnbr` | Non-blocking send, blocking receive (anti-deadlock variant) |
| `_man` | Manual / hand-rolled implementation (no collective primitive) |
| `_comm_only` | Communication-only variant (no computation, C++) |
| `_get` / `_put` | One-sided (RMA) variant |

### Collectives

#### All-to-all — `alltoall/`
Every rank sends a distinct message to every other rank.

| Binary | MPI primitive |
|--------|--------------|
| `alltoall_b` | `MPI_Alltoall` |
| `alltoall_nb` | `MPI_Ialltoall` |
| `alltoall_man` | `MPI_Isend` / `MPI_Irecv` (manual) |
| `alltoall_comm_only` | `MPI_Isend` / `MPI_Irecv` (C++, no memory init) |

#### All-gather — `allgather/`
Every rank broadcasts its buffer; all ranks collect the full result.

| Binary | MPI primitive |
|--------|--------------|
| `allgather_b` | `MPI_Allgather` |
| `allgather_nb` | `MPI_Iallgather` |
| `allgather_comm_only` | `MPI_Isend` / `MPI_Irecv` (C++) |

#### All-reduce — `allreduce/`
Global reduction (`MPI_SUM` over `int`) delivered to all ranks.

| Binary | MPI primitive |
|--------|--------------|
| `allreduce_b` | `MPI_Allreduce` |
| `allreduce_nb` | `MPI_Iallreduce` |

#### Barrier — `barrier/`

| Binary | MPI primitive |
|--------|--------------|
| `barrier_b` | `MPI_Barrier` |
| `barrier_nb` | `MPI_Ibarrier` |

#### Broadcast — `broadcast/`
Root sends one buffer to all other ranks.

| Binary | MPI primitive |
|--------|--------------|
| `broadcast_b` | `MPI_Bcast` |
| `broadcast_nb` | `MPI_Ibcast` |

#### Gather — `gather/`
All ranks send to root; root collects all messages.

| Binary | MPI primitive |
|--------|--------------|
| `gather_b` | `MPI_Gather` |
| `gather_nb` | `MPI_Igather` |

#### Scatter — `scatter/`
Root distributes a distinct chunk to every rank.

| Binary | MPI primitive |
|--------|--------------|
| `scatter_b` | `MPI_Scatter` |
| `scatter_nb` | `MPI_Iscatter` |

#### Reduce — `reduce/`
Global reduction (`MPI_SUM` over `int`) to root only.

| Binary | MPI primitive |
|--------|--------------|
| `reduce_b` | `MPI_Reduce` |
| `reduce_nb` | `MPI_Ireduce` |

#### Reduce-scatter — `reduce_scatter/`
Reduction followed by a scatter of the result chunks.

| Binary | MPI primitive |
|--------|--------------|
| `reduce_scatter_b` | `MPI_Reduce_scatter` |
| `reduce_scatter_nb` | `MPI_Ireduce_scatter` |

---

### Point-to-point patterns

#### Ping-pong — `pingpong/`
Latency measurement between a pair of ranks.

| Binary | Description |
|--------|------------|
| `pingpong_b` | Single pair (rank 0 ↔ rank 1) |
| `pingpong_pairwise_b` | All pairs simultaneously |

#### Incast — `incast/`
All non-root ranks send to root simultaneously.

| Binary | Variant |
|--------|---------|
| `incast_b` | Blocking receive at root |
| `incast_nb` | Non-blocking (`MPI_Irecv` at root) |
| `incast_bsnbr` | Non-blocking send, blocking receive |
| `incast_get` | One-sided: root issues `MPI_Get` from each sender |
| `incast_put` | One-sided: senders issue `MPI_Put` to root window |

#### Pairwise — `pairwise/`
Each rank exchanges messages with exactly one partner.  Partners are chosen by offset or at random.

| Binary | Variant |
|--------|---------|
| `pairwise_b` | Blocking (`MPI_Sendrecv`) |
| `pairwise_nb` | Non-blocking (`MPI_Isend` / `MPI_Irecv`) |
| `pairwise_bsnbr` | Non-blocking send, blocking receive |

Extra flags: `-offset <O>` (default `n/2`, half the number of ranks) — fixed partner offset; `-mode <M>` (default `offpair`) — pairing mode. Supported modes:

| Mode | Meaning |
|------|---------|
| `offpair` | Disjoint reciprocal pairs at fixed offset `O` (e.g. `(0,1) (2,3) …` for `O=1`) |
| `rpair` | Uniformly random disjoint pairs (same seed on all ranks) |
| `perm` | Random permutation: each rank sends to one partner (not necessarily reciprocal) |
| `rot` | Ring rotation: rank `i` sends to `(i+O) mod n` |

#### Ring — `ring/`
Each rank simultaneously sends to both its left and right neighbours and receives one message from each (bidirectional ring exchange).

| Binary | Variant |
|--------|---------|
| `ring_nb` | Non-blocking (`MPI_Isend` / `MPI_Irecv`) |
| `ring_bsnbr` | Non-blocking send, blocking receive |

Extra flag: `-rring` — randomise the ring order instead of using rank order.

#### Tag matching — `tagmatch/`

Two ranks exchange `N` messages per iteration with independently chosen tag orderings.  The benchmark is meant to expose differences in how an MPI implementation matches messages to receives.

To bridge sends and receives that do not happen simultaneously, an implementation may conceptually maintain two queues:

- an **Unexpected Message Queue (UMQ)** — messages that arrived before a matching `Recv` was posted, drained when a `Recv` is finally issued;
- a **Posted Receive Queue (PRQ)** — `Irecv`s posted before the matching message arrived, drained as messages come in.

If the queues are scanned linearly, sending and receiving `N` tags in the **same** order finds each match near the head — roughly `O(N)` total scan work — whereas using **opposite** orders could in principle force each match to walk to the tail of a growing queue — up to `O(N²)`.  Whether a given implementation actually behaves this way depends on its data structures; this benchmark only sets up the conditions under which such a difference would become visible, and lets you compare the two queues separately.

Binary: `tagmatch_nb` (2 ranks)

Extra flags:

| Flag | Default | Description |
|------|---------|-------------|
| `-ntags <N>` | `1024` | Number of messages per iteration (capped at `MPI_TAG_UB+1`) |
| `-sendorder <inc\|dec\|random\|same>` | `inc` | Order in which the sender issues tags. `same` is an alias for `inc`. |
| `-recvorder <inc\|dec\|random\|same>` | `dec` | Order in which the receiver matches tags |
| `-prepost` | off | Receiver pre-posts all `Irecv`s (targets the PRQ-matching path). Default off = post-then-Wait per message (targets the UMQ-matching path). |
| `-wildcard` | off | Receiver uses `MPI_ANY_TAG` — exercises the wildcard match code path. |

```bash
# Opposite tag orders, no pre-posting — exercises the UMQ-matching path:
mpirun -n 2 build/bin/tagmatch_nb -ntags 4096 -iter 10

# Same scenario but with all Irecvs pre-posted — exercises the PRQ path:
mpirun -n 2 build/bin/tagmatch_nb -ntags 4096 -iter 10 -prepost

# Visualise per-iteration latencies; compare runs with -recvorder dec
# vs -recvorder inc to see whether the implementation shows different
# scaling behaviour between the two orderings:
mpirun -n 2 build/bin/tagmatch_nb -ntags 16384 -iter 200 -plot -plotlog
```

Note: `-blength` / `-bpause` are accepted but have no effect — the benchmark has no inner-loop burst structure.

---

### Synthetic / application-inspired patterns

#### 2-D stencil — `stencil/`
Each rank exchanges halo data with its four Cartesian neighbours (north, south, west, east).  Uses an `MPI_Cart_create` communicator.  Boundary ranks have `MPI_PROC_NULL` neighbours which complete immediately with no data transfer, so no special-casing is needed for edge ranks.

Binary: `stencil_2d_nb`

Extra flags:

| Flag | Default | Description |
|------|---------|-------------|
| `-dimx <X>` | `0` | Fix the X dimension of the process grid; Y is derived as `w_size / X`.  `0` lets `MPI_Dims_create` choose automatically. |
| `-periodic` | off | Make the Cartesian grid periodic (toroidal) in both dimensions |

```bash
mpirun -n 16 build/bin/stencil_2d_nb -msgsize 65536 -iter 100 -dimx 4
```

#### k-random partners — `kpartners/`
Each rank communicates with *k* randomly chosen partners per iteration.  The partner sets are derived from *k* independent random permutations (same seed on all ranks), guaranteeing that the resulting graph is k-regular: every rank sends exactly *k* messages and receives exactly *k* messages.

Binary: `kpartners_nb`

Extra flag:

| Flag | Default | Description |
|------|---------|-------------|
| `-k <K>` | `1` | Number of random partners per rank (capped at `w_size - 1`) |

```bash
mpirun -n 32 build/bin/kpartners_nb -k 4 -msgsize 4096 -iter 200
```

---

## Output format

### CSV (default)

All benchmarks print a CSV block to stdout on the master rank.  Each line corresponds to one measured iteration:

```
Average,Minimum,Maximum,Median,MainRank
<avg>,<min>,<max>,<median>,<master_rank_latency>
...
Ran N iterations. Measured M iterations.
```

Times are in seconds (9 decimal places).  The `Average`, `Minimum`, `Maximum`, and `Median` columns are computed across all MPI ranks for that iteration.  `MainRank` is the raw value recorded on the master rank.

### Pretty-print

Pass `-pretty-print` to get a human-readable table with auto-scaled units (`us` / `ms` / `s`) instead of CSV:

```bash
mpirun -n 8 build/bin/alltoall_nb -iter 5 -pretty-print
```

```
     #          avg          min          max       median
  -------------------------------------------------------------
      1      12.45 us      9.12 us     18.67 us     11.34 us
      2      11.98 us      9.08 us     17.23 us     11.12 us
  -------------------------------------------------------------
  5 samples · 10 iterations total
```

The `MainRank` column is omitted in pretty mode.  Pipe through a tool such as `sed 's/\x1b\[[0-9;]*m//g'` to strip ANSI colour codes if needed.

### Distribution plot

Pass `-plot` to **append** an ASCII histogram of the per-iteration timing distribution (the CSV / pretty table is still printed first).  Each iteration contributes one value — the cross-rank statistic chosen with `-plotstat` (default `max`, i.e. the slowest rank, which is the completion time of a collective).  A percentile footer summarises the tail regardless of how the bins fall.

```bash
mpirun -n 8 build/bin/alltoall_nb -iter 400 -plot -plotstat max
```

```
  Per-iteration latency  ·  stat=max  ·  n=400  ·  linear bins
  ----------------------------------------------------------------------
  [     6.05 us -     32.88 us]  ████████████████████████████████████████   97.8%
  [    32.88 us -     59.70 us]                                              1.0%
       ...
  [   247.49 us -    274.31 us]                                              0.2%
  ----------------------------------------------------------------------
  n=400 · min 6.05 us · mean 9.09 us · p50 6.62 us · p90 7.15 us · p99 74.71 us · max 274.31 us
```

Binning options:

- `-plotbins <N>` — number of bins (default `10`); always produces a readable, fixed-height histogram.
- `-plotbinsize <D>` — fixed bin **width** with an optional unit (`2ms`, `500us`, `100ns`, or a bare number = seconds).  Edges are aligned to multiples of the width so plots from different runs line up.  Overrides `-plotbins`; capped at 64 bins (the tail folds into the last bin, with a note) so a tiny width can't flood the terminal.
- `-plotlog` — logarithmic (geometric) bins, which reveal the shape of heavy-tailed latency distributions far better than linear bins.  Cannot be combined with `-plotbinsize`.

As with pretty-print, pipe through `sed 's/\x1b\[[0-9;]*m//g'` to strip the ANSI colours.

## Early termination

Send `SIGUSR1` to the master process to trigger a clean shutdown: remaining iterations are skipped, results collected so far are printed, and all ranks call `MPI_Finalize`.

```bash
kill -USR1 <pid_of_rank_0>
```
