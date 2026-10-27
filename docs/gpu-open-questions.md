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
  (`local_rank % ngpus`) and print one warning from rank 0. Alternative: refuse to run.

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

**To ask:** on the target clusters' MPI, is one-sided communication on GPU buffers expected to work (and with which
MPI/UCX settings)? Should those two benchmarks simply be Host-mode-only?

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
