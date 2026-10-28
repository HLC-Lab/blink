# Porting a Benchmark to `-gpu`

A Benchmark keeps its host buffers and host fill/verify loops. MPI is handed **device mirrors** of those buffers
(`data_mirror`). In Host mode a mirror is the host pointer itself, so the host path is unchanged.

1. Keep `send_buf`/`recv_buf` as host buffers, allocated with `malloc_align` and filled exactly as today. Keep the
   existing host verification loops unchanged. Keep `durations` on the host.
2. Right after the host fill, and before the first MPI call that uses the buffer, add device mirrors and name them
   `<buf>_dev`:
   ```c
   unsigned char *send_dev = (unsigned char*)data_mirror(send_buf, send_buf_size);
   unsigned char *recv_dev = (unsigned char*)data_mirror(recv_buf, recv_buf_size);
   ```
   (use the buffer's real element type and byte size).
3. Inside the timed loop, replace the buffer arguments of every MPI call (`MPI_Send/Recv/Isend/Irecv/Sendrecv/
   Alltoall/Allreduce/.../Win_create/Put/Get`) with the `_dev` pointers. Change nothing else in the loop.
4. If the benchmark rewrites a buffer's *contents* on the host between iterations (tags, per-iteration refills),
   call `data_push(<buf>_dev, <buf>, size)` after the host write, outside the timed region. If it ever swaps or
   reassigns the host buffer *pointers*, swap the `_dev` pointers in lockstep at the same place (no `data_push`:
   pushing would upload stale host data), and fetch and free by the final pairing. (No benchmark in the suite
   swaps pointers today; `ring_*` only takes `recv_buf + msg_size` sub-pointers, which map to `recv_dev + msg_size`.)
5. At the start of the `if (debug_mode)` block, before the verify loops, add
   `data_fetch(recv_buf, recv_dev, recv_buf_size);` (and the same for any other buffer the verify loop reads). If
   the debug block itself runs further MPI exchanges (e.g. `pairwise_b`), fetch after them, right before the check.
6. Next to the existing `free(...)` of the buffers (after `done:`), add `data_free(send_dev); data_free(recv_dev);`
   before `MPI_Finalize`.
7. Host-only `memcpy` self-copies of Data buffers (the `*_comm_only.cpp` files) become
   `data_copy(dst_dev, src_dev, n)`.
8. In-place buffers (one buffer both sent and received): one mirror, one fetch, one free.
9. If an MPI window is created over a buffer, create it over the `_dev` pointer.
10. Do not add `#include "gpu.h"`; benchmarks only use `common.h`.

## How a port is checked

Two independent checks; **both** are required, neither is enough alone.

1. **Runtime (`_gpu` twins, `BLINK_TEST_GPU=1`).** Every run gets `-gpu`, and `run_debug` fails when a rank reports
   `gpu=1` but created no mirror (`mirrors=0` in its `DEBUG` line). A *receive*-side slip (MPI writes the host
   buffer, then `data_fetch` overwrites it with the untouched device mirror) shows up as `check=FAIL`. A *send*-side
   slip does **not**: host and device copies hold the same bytes, receivers still verify OK, `mirrors` is still
   nonzero, and MPI is silently reading host memory.
2. **Static gate.** `python3 -I $SCRATCH/gate.py <files>` flags any `MPI_*(...)` call whose arguments contain a host
   Data buffer that the file also mirrors (use the `_dev` pointer), prints `REVIEW:` lines for non-MPI calls that
   take a host buffer (the `*_comm_only.cpp` algorithm functions must receive `_dev` pointers), and lists the
   `malloc_align`'d names that are *not* mirrored so a human confirms they are host-only (`durations`, requests,
   `targets`, tag arrays). Must exit 0 with no `REVIEW:` line that is not explained.

## Worked example: `src/alltoall/alltoall_b.c`

```diff
         send_buf[bi] = debug_mode ? (unsigned char)my_rank : 'a';
     }
+
+    /*GPU mode: MPI works on device mirrors; in host mode the mirrors are the host buffers*/
+    unsigned char *send_dev = (unsigned char*)data_mirror(send_buf, send_buf_size);
+    unsigned char *recv_dev = (unsigned char*)data_mirror(recv_buf, recv_buf_size);
 ...
-                    MPI_Alltoall(send_buf,msg_size,MPI_BYTE,recv_buf,msg_size,MPI_BYTE,MPI_COMM_WORLD);
+                    MPI_Alltoall(send_dev,msg_size,MPI_BYTE,recv_dev,msg_size,MPI_BYTE,MPI_COMM_WORLD);
 ...
     if (debug_mode) {
+        data_fetch(recv_buf, recv_dev, recv_buf_size);
         int _r, _b, _ok = 1;
 ...
     free(send_buf);
     free(recv_buf);
+    data_free(send_dev);
+    data_free(recv_dev);
```

A point-to-point example with a post-loop debug exchange and an offset into the receive buffer is
`src/pairwise/pairwise_b.c` (`&recv_dev[i*msg_size]`, fetch after the debug exchange).
