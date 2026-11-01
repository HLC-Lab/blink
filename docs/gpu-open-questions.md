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

**Professor's answer:** OK, the automatic mapping (wrap-around with a single warning) is fine as it is.

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

**Professor's answer:** OK, the timed region is as described: only the MPI call is measured, the host-to-device
and device-to-host copies stay outside it.

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

**Professor's answer:** not asked yet.  Leaning towards yes (available in GPU mode only where the probe passes).

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

- **Lighter CI (done, 2026-11-01).**  The `compile-only (CUDA)` job used to install the whole `nvidia-cuda-toolkit`
  apt package (several GB; about 2 minutes on one run, over 15 on another).  The GPU compile jobs now live in
  `.github/workflows/gpu-build.yml`, which only runs when `src/`, `CMakeLists.txt`, `CMakePresets.json` or the workflow
  itself change, and the CUDA job installs `cuda-cudart-dev-12-9` plus `cuda-nvcc-12-9` from NVIDIA's apt repo (about
  40 MB).  `nvcc` is needed: with the runtime package alone, `FindCUDAToolkit` reports no toolkit.  The whole job takes
  about 50 seconds.

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
| Large-message offsets (2026-11-01, `main` at `ba27cf03`): the opt-in `test_large` (8 cases, 1 node, host buffers) | all 8 cases ran and passed, none skipped |
| `alltoall_comm_only` and `alltoall_man`, 8 ranks over 2 nodes, `-msgsize` 512 MiB and 1 GiB, `-iter 3 -warmup 1 -debug`, Host and `-gpu` | exit 0 and `check=OK` on all 8 ranks in all 8 runs |

The MPI reports GPU support (`MPI reports GPU support: yes`); UCX was configured with CUDA and gdrcopy.

### Leonardo follow-up checks (2026-11-01)

**UCX context on device 0: not a problem here.**  4 ranks of `alltoall_b -gpu -endl` on one node with all 4 GPUs
visible: `nvidia-smi --query-compute-apps` showed exactly one process per GPU (482 MiB each) and no extra process on
GPU 0, so UCX does not create a context on device 0 before `gpu_set_device` runs (Open MPI 4.1.6 + UCX 1.16).  No
change needed; selecting the device before `MPI_Init` is not required on this stack.  The `--gpus-per-task=1` control
did not run (the first step's tasks outlived the `SIGINT` used to stop them), but it could not show an extra context.

**Open MPI 5.0.9 (`openmpi/5.0.9--gcc--12.2.0-cuda-12.2`): nonblocking reductions could not be tested.**  It builds
Blink fine, but `srun` launches time out with both `--mpi=pmix` and `--mpi=pmix_v5` (`Socket timed out on send/recv
operation`, even for a host-buffer `barrier_b`).  `mpirun` inside the allocation launches, and then every `-gpu` run
(with and without `OMPI_MCA_coll_ucc_enable=1`) stops at Blink's own startup probe: "this MPI is not GPU-aware".  So
the nonblocking reductions are never reached.  Possible cause, not verified: this module's UCX/UCC come from HPC-X 2.25.1
under `nvhpc-25.11` (`comm_libs/13.0`), while the loaded CUDA is 12.2.  Left as it is: use Open MPI 4.1.6 on Leonardo.

**The 1.9x GPU-vs-Host gap between nodes is a single NIC lane (Open MPI 4.1.6 + UCX 1.16, 2 nodes, 1 rank each,
`pingpong_b`, 200 iterations, mean of the Average column, one-way).**

| Message | Host, default | Host, 1 lane | GPU, default | GPU, no GPUDirect RDMA |
|---|---|---|---|---|
| 8 B | 0.9 us | 1.0 us | 2.8 us | 2.7 us |
| 128 KiB | 15.1 us | 15.4 us | 15.8 us | 15.8 us |
| 2 MiB | 93.0 us | 177.7 us | 177.7 us | 177.5 us |
| 64 MiB | 2727 us (24.6 GB/s) | 5445 us (12.3 GB/s) | 5446 us (12.3 GB/s) | 4610 us (14.6 GB/s) |

- From about 512 KiB up, host buffers reach twice the bandwidth of GPU buffers.  Pinning host buffers to one NIC
  (`UCX_NET_DEVICES=mlx5_0:1`) or to one rendezvous lane (`UCX_MAX_RNDV_LANES=1`) gives exactly the GPU numbers, so the
  GPU path moves data over one lane (about one HDR100 NIC, 12.3 GB/s) where host buffers use two.
- Giving the GPU path more room changes nothing: `UCX_MAX_RNDV_LANES=4`, `UCX_NET_DEVICES=mlx5_0:1,mlx5_1:1` and all
  four NICs all stay at 5444 us for 64 MiB.  `UCX_RNDV_SCHEME=get_zcopy` / `put_zcopy` also change nothing.
- `UCX_IB_GPU_DIRECT_RDMA=no` (staging through the host) is faster for large GPU messages (14.6 GB/s) but not
  two-rail either.  A NIC that is not the one next to the GPU is slower for GPU buffers (6139 us at 64 MiB with
  `mlx5_1`), so the NIC choice matters, and the default one is the right one.
- Small messages: GPU buffers cost about 1.9 us more than host buffers (2.8 vs 0.9 us at 8 B); from 8 KiB to 128 KiB
  the two are within about 1 us.
- Node topology (`nvidia-smi topo -m`): GPU *i* is `PXB` to `mlx5_i` and `NODE` to the other three NICs, so each GPU has
  one NIC close to it.  A likely reason for the single lane, not verified: only the near NIC qualifies for GPUDirect
  RDMA from that GPU.
- Within a node, GPU buffers are much faster than host buffers (724 us vs 8726 us at 64 MiB, NVLink vs shared memory).
- So the gap is how this UCX moves GPU memory, not Blink: the timed region contains only the MPI call.

A question for the professor / cluster support: is a single lane for GPU memory the expected behaviour on Leonardo with
this UCX, and is there a setting (or a newer HPC-X) that lets GPU buffers use more than one NIC?

**Professor's answer:** not asked yet; more timing tests are planned first.

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

Checked on Leonardo (2026-11-01): UCX does not create an extra context on device 0 (see "Leonardo follow-up checks"
above), so the device is not selected before `MPI_Init`.  Not tested on other stacks.

Observation (stderr redirection): `mpirun -n 2 <bin> -gpu 2>&1 >/dev/null | head` showed rank stdout lines in the
pipe (the launcher's own I/O forwarding interacts with that redirection order), while `mpirun ... >out 2>err` split
the two streams correctly. Wrappers should redirect to files (or use the launcher's own output options), not rely on
`2>&1 >/dev/null` ordering.
