# GPU port: open questions for the professor

Design choices made during planning the `-gpu` mode (see `CONTEXT.md` for the terms). Each one is a working
assumption to validate with the professor after implementation and testing.

## 1. Rank-to-GPU mapping

**Current assumption:** automatic. After `MPI_Init`, each rank computes its node-local rank
(`MPI_Comm_split_type(SHARED)`) and uses device `local_rank % visible_device_count`. When a launcher restricts
visibility (Slurm GPU binding, `CUDA_VISIBLE_DEVICES`), each rank sees one device and uses device 0.

**To ask:**
- a) Would a direct, user-chosen rank-to-GPU mapping (e.g. a `-gpumap` style flag) be useful for the experiments, or
  is the automatic mapping good enough?
- b) What should happen when ranks per node exceed GPUs per node? Current assumption: wrap around
  (`local_rank % ngpus`) and print one warning from rank 0 when ranks really share a physical GPU (compared by PCI
  id, so one-GPU-per-rank launcher binding does not warn). Alternative: refuse to run.

## 2. Where the timed region starts in GPU mode

**Current assumption:** the timed loop is unchanged and measures only the MPI call (`MPI_Wtime` around it, as in
Host mode). All GPU-specific work happens outside it:
- Before the loop: fill a host staging buffer, then copy it to the device buffer.
- After the loop, only under `-debug`: copy the receive buffer back to the host, then verify on the host.
- No kernels, so no device synchronisation is needed inside the timed region.

**To ask:**
- Is this the right flow, i.e. should timing start after the initialisation and host-to-device copy, exactly as
  Host mode starts it after the buffer fill?
- Should the host-to-device and device-to-host copies ever be part of what we measure (they are not now)?

## 3. One-sided communication on GPU memory (`incast_get`, `incast_put`)

**Finding (2026-10-27, this laptop: Open MPI 5.0.11 + UCX, one RTX 3060, 2 and 8 ranks):** point-to-point
(`MPI_Sendrecv`) and `MPI_Allreduce` on device pointers are correct. `MPI_Win_create` + `MPI_Put` on device memory
returns without any error but the data does not arrive (verified against an identical host-buffer run that works).
`--mca osc ucx` gives the same result, and `--mca osc rdma` aborts inside `MPI_Win_create`.

**Current assumption:** `-gpu` on these two benchmarks first runs a one-sided probe and aborts with a clear message
if the data does not arrive, instead of reporting meaningless timings.

**Leonardo (2026-10-28, Open MPI 4.1.6 + UCX 1.16 from HPC-X, CUDA 12.2, A100 nodes, 8 ranks over 2 nodes):** the
probe passes and both benchmarks run on device memory with `check=OK` on every rank.  So one-sided on GPU buffers works
there; only the laptop's Open MPI 5.0.11 + UCX loses the data.  The probe stays as a safeguard for MPIs like that one.

**To ask:** nothing blocking any more.  For the record: should these two stay available in GPU mode only where the probe
passes (current behaviour), or be documented as supported only on specific MPI/UCX stacks?

## 4. Nonblocking reductions on GPU memory (`allreduce_nb`, `reduce_nb`, `reduce_scatter_nb`)

**Finding (2026-10-27, same machine):** the blocking reductions work on device pointers, but `MPI_Iallreduce`,
`MPI_Ireduce` and `MPI_Ireduce_scatter` segfault inside Open MPI's libnbc (the reduction operator is applied on the
CPU to a device address).

**Current assumption:** same policy as one-sided: `-gpu` on these three benchmarks runs a tiny probe (with a
SIGSEGV/SIGBUS handler) and aborts with an explicit "not supported by this MPI" message. On an MPI that supports them
(e.g. with a GPU-aware libnbc/UCC) the probe passes and the benchmarks run.

**Leonardo (2026-10-28, Open MPI 4.1.6 + UCX 1.16, 8 ranks over 2 nodes):** same result as the laptop: the probe
catches the crash and `-gpu` stops with the explicit message on all three.  So it is not specific to one Open MPI
version or machine.

**To ask:** is it acceptable that the three `_nb` reduction benchmarks are unavailable in GPU mode (they work in Host
mode, and the blocking `allreduce_b`/`reduce_b`/`reduce_scatter_b` work in GPU mode), or should they get a different
path (e.g. UCC, or another Open MPI version)?

## TODO

- **Lighter CI.**  The `compile-only (CUDA)` job installs the whole `nvidia-cuda-toolkit` apt package (several GB; it
  took about 2 minutes on one run and over 15 on another, on a docs-only commit).  Options: install only the CUDA
  runtime headers/`libcudart` (e.g. `nvidia-cuda-toolkit` replaced by the smaller `cuda-cudart-dev` from NVIDIA's
  repo), cache the apt packages, and run the GPU compile jobs only when `src/`, `CMakeLists.txt` or `.github/` change
  (a `paths:` filter).  Not started.

## Known limitations (not for the professor)

- `gpu_setup()` runs inside `parse_common_args`, so the GPU startup probes run before a benchmark validates its own
  flags (rank count for `pingpong_*`/`tagmatch_nb`, `-k` for `kpartners_nb`).  A bad flag therefore aborts about a
  second later than it would in Host mode.  The probes are untimed.  Not changed: it would touch every benchmark.

## Leonardo validation (2026-10-28)

Open MPI 4.1.6 (CUDA 12.2) + UCX 1.16, 1 node with 4 A100 (64 GB) and 2 nodes, run through `srun` in a `salloc`.

| Check | Result |
|---|---|
| Test suite on real GPUs (9 `_gpu` twins, `test_gpu`, `test_gpu_layer`) | passed; 3 tests had environment assumptions that do not hold under Slurm / Open MPI 4 and now skip (see git log) |
| 4 ranks, 4 visible GPUs | `device=0..3`, no sharing warning |
| 4 ranks, `srun --gpus-per-task=1` (each rank sees 1 GPU) | `device=0` on every rank, **no sharing warning** (PCI-id check) |
| 8 ranks on 4 GPUs | `device=0..3,0..3`, sharing warning exactly once |
| `incast_get`, `incast_put` | work on device memory, `check=OK` (1 KB and 1 MiB, 8 ranks over 2 nodes) |
| `allreduce_nb`, `reduce_nb`, `reduce_scatter_nb` | rejected with the explicit message |
| Inter-node, 1 MiB: `pingpong_b` (2 ranks), `alltoall_b` (8 ranks) | `check=OK` on every rank |
| Timing, `pingpong_b` 1 MiB, 2 nodes, 1 rank each, 200 iterations, mean of the Average column | Host 48.8 µs, `-gpu` 91.2 µs (1.9x); an earlier `-debug` run gave the same 89 to 90 µs, so it is reproducible |

The MPI reports GPU support (`MPI reports GPU support: yes`); UCX was configured with CUDA and gdrcopy.

Not verified on Leonardo: whether UCX creates a CUDA context on device 0 for every rank before `gpu_set_device` runs.
The `nvidia-smi --query-compute-apps` check was tried twice and listed no processes both times: the 4-rank
`alltoall_b` finished (about 100 us per iteration on NVLink) before `nvidia-smi` ran.  Retry with a much larger `-iter`
(e.g. 2000000) or the endless mode `-endl`.  Device selection itself is
confirmed: each rank used the device it was assigned and all data checks passed.  The possible cost is a small
extra context per rank on GPU 0, not wrong results.  Also not compared: Host vs GPU timings, and Open MPI 5 on
Leonardo (`openmpi/5.0.9--gcc--12.2.0-cuda-12.2`) for the nonblocking reductions (deferred).

The 1.9x GPU-vs-Host gap at 1 MiB between nodes is a property of how this MPI/UCX moves device memory (Blink only
passes device pointers, and the timed region contains the MPI call alone), e.g. staging through the host or a NIC not
matched to the GPU.  The cause was not investigated.  A question for the professor / cluster support: is that gap
expected on Leonardo, and which UCX settings (`UCX_NET_DEVICES`, GPUDirect RDMA, rail choice) are recommended?

## Checks for the final Opus review (not for the professor)

### stderr output and the CRAB wrappers

**Current assumption:** every GPU-mode message (the "`-gpu` has no effect" warning for Buffer-less executables, the
rank-oversubscription warning, the GPU-aware-MPI query line) goes to stderr, once, from rank 0. Stdout stays
untouched, because it carries the CSV and the `DEBUG` lines that `tests/` parse.

**To check:** look at the neighbouring directories (`../crab-wrappers`, `../CRAB`) for how they launch Blink
executables and what they do with stdout and stderr. Confirm that:
- the wrappers don't merge stderr into stdout (`2>&1`, `srun --output` merging, or a combined log) and so corrupt
  the CSV;
- they don't treat any stderr output as a failure;
- the Blink test harness (`popen` in `tests/popen_helpers.h`) only reads stdout, or tolerates stderr;
- the `-gpu` flag passes through the wrappers' argument handling and doesn't clash with a flag of their own.

**Result (2026-10-27):** checked, no problem on either side.
- CRAB builds `<mpirun|srun ...> <binary> <args>` (`CRAB/src/crab/wrappers/base.py`) and runs it with stdout and
  stderr as separate pipes (`core/process/manager.py`); stderr is only copied to a log, never merged into stdout.
- The Blink wrappers (`crab-wrappers/blink/_microbench_common.py`, `read_data`) parse only stdout, starting at the
  CSV header, and skip lines that don't start with a digit.
- A run fails only on a non-zero exit code; stderr is then written to the failure log, so a "not supported by this
  MPI" abort shows up there.
- `args` is passed verbatim, so `-gpu` reaches the binary; no wrapper has a GPU flag or sets `CUDA_VISIBLE_DEVICES`.
- `null_dummy` is documented in `crab-wrappers/blink/README.md` as ignoring its arguments, so `null_dummy` and
  `dist_test` keep ignoring every argument except `-gpu`.
- A pre-existing CRAB-side risk unrelated to `-gpu`: stderr is a pipe that is read only after the process exits, so
  more than about 64 KB of stderr (a very chatty UCX/CUDA stack) would block the run until its timeout.

### Branch lifecycle

**Current assumption:** all work happens on a `gpu` branch (GPU layer plus a pilot first, then one commit per
benchmark group). Once everything is clean, tested and reviewed, merge it into `main` and delete the `gpu` branch.
Nothing is pushed or merged without an explicit go-ahead.

**To check:** at the end, confirm the full-diff review, the local GPU test run and the compile-only CI job are all
green before merging.

### Device selection vs. UCX context creation

On multi-GPU nodes UCX may create its CUDA context on device 0 during `MPI_Init`, before `gpu_set_device` runs.
Whether the device should be selected before `MPI_Init` (from the launcher's local-rank environment variable, e.g.
`OMPI_COMM_WORLD_LOCAL_RANK` / `SLURM_LOCALID`) cannot be tested on this single-GPU machine. Check on a multi-GPU
node, or with the professor.

Observation to include in that check: `mpirun -n 2 <bin> -gpu 2>&1 >/dev/null | head` showed rank stdout lines in the
pipe (the launcher's own I/O forwarding interacts with that redirection order), while `mpirun ... >out 2>err` split
the two streams correctly. Wrappers should redirect to files (or use the launcher's own output options), not rely on
`2>&1 >/dev/null` ordering.
