# GPU mode uses GPU-aware MPI only, with no GPU kernels

`-gpu` runs the same MPI calls on device pointers (GPU-aware MPI) rather than switching to NCCL/RCCL, which cannot
express tags, wildcard receives, one-sided operations or barriers and would change what the timers measure. GPU code
is limited to a thin allocate/free/copy wrapper (CUDA or HIP) with host staging for init and `-debug` verification, so
the suite builds with plain gcc and needs no nvcc, hipcc or per-vendor kernels.
