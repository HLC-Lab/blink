# Blink

An MPI benchmark suite that measures the time of communication patterns (collectives, point-to-point, one-sided) under controlled conditions, including bursty load.

## Language

**Benchmark**:
An executable that runs one communication pattern repeatedly and reports per-iteration timings.
_Avoid_: test, microbenchmark

**Tool**:
An executable in the suite that is not a Benchmark (`checker`, `dist_test`, `null_dummy`).
_Avoid_: utility, helper

**Data buffer**:
The send or receive memory a Benchmark passes to MPI.
_Avoid_: payload, message memory

**GPU mode**:
A run, selected with `-gpu`, in which a Benchmark's Data buffers live in GPU memory and MPI communicates with them directly.
_Avoid_: CUDA mode, device mode

**Host mode**:
The default run, in which Data buffers live in ordinary host memory.
_Avoid_: CPU mode

**GPU-aware MPI**:
An MPI implementation that accepts GPU memory pointers in communication calls. It is the only communication layer GPU mode uses.
_Avoid_: CUDA-aware MPI (names one vendor only)

**GPU backend**:
The vendor runtime that provides GPU memory (CUDA for NVIDIA, HIP for AMD).
_Avoid_: GPU library, driver

**Buffer-less executable**:
A Benchmark or Tool with no Data buffers, so GPU mode has no effect on it.
_Avoid_: no-op benchmark
