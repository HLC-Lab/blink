/*
 * tagmatch_nb.c — exercises MPI tag matching with N messages per iteration.
 *
 * Two ranks exchange N tagged messages.  Conceptually, an MPI implementation
 * may maintain two queues to bridge the gap between sends and recvs that do
 * not happen simultaneously:
 *   - UMQ (Unexpected Message Queue): messages that arrived before a matching
 *     recv was posted.  Drained when the receiver issues Recv/Irecv.
 *   - PRQ (Posted Receive Queue): Irecvs posted before the matching message
 *     arrived.  Drained when messages arrive.
 *
 * If a particular implementation scans either queue in linear time, sending
 * and receiving N tags in opposite orders could in principle cost N*(N+1)/2
 * scan steps (O(N^2)) instead of O(N) when the orders match.  Whether any
 * specific implementation actually exhibits this depends on its data
 * structures; this benchmark only sets up the conditions under which such a
 * difference would become visible.
 *
 * Modes:
 *   -prepost off (default) — receiver does Irecv-then-Wait per message;
 *       while it waits for the current tag, other messages may accumulate
 *       in the UMQ.  Targets the UMQ-matching path.
 *   -prepost on            — receiver pre-posts all N Irecvs (which may
 *       end up in the PRQ) and then Waits on them all.  Targets the
 *       PRQ-matching path.
 *
 * Always uses Irecv (Recv is equivalent to Irecv+Wait from the matching
 * engine's perspective).
 *
 * Requires exactly 2 ranks.
 */
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <limits.h>
#include "common.h"

const char *benchmark_help =
"  -ntags <N>                    messages per iteration (default 1024, capped at MPI_TAG_UB+1)\n"
"  -sendorder <inc|dec|random|same>  sender's tag order (default: inc; 'same' = inc)\n"
"  -recvorder <inc|dec|random|same>  receiver's tag order (default: dec; 'same' = inc)\n"
"  -prepost                      receiver pre-posts all Irecvs (targets PRQ-matching path)\n"
"                                default off: post-then-Wait per message (targets UMQ-matching path)\n"
"  -wildcard                     receiver uses MPI_ANY_TAG (exercises the wildcard match path)\n";

/* Build a tag-order permutation of [0..n-1] into `out`. Aborts on bad name. */
static void build_order(int *out, int n, const char *order, const char *flag)
{
    int i;
    for (i = 0; i < n; i++) out[i] = i;
    if (strcmp(order, "inc") == 0 || strcmp(order, "same") == 0) {
        /* already 0..n-1 */
    } else if (strcmp(order, "dec") == 0) {
        for (i = 0; i < n / 2; i++) {
            int t = out[i]; out[i] = out[n-1-i]; out[n-1-i] = t;
        }
    } else if (strcmp(order, "random") == 0) {
        /* Fisher-Yates; both ranks use same RNG seed, so they produce the
         * same permutation independently — that's fine, the benchmark cares
         * about send vs recv order *relative*, not absolute. */
        for (i = n - 1; i > 0; i--) {
            int j = rand() % (i + 1);
            int t = out[i]; out[i] = out[j]; out[j] = t;
        }
    } else {
        if (my_rank == master_rank)
            fprintf(stderr, "%s: unknown order '%s' (use inc|dec|random|same)\n", flag, order);
        MPI_Abort(MPI_COMM_WORLD, -1);
    }
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &w_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    install_shutdown_handler();

    int ntags = 1024;
    const char *sendorder = "inc";
    const char *recvorder = "dec";
    bool prepost  = false;
    bool wildcard = false;

    int i, k;
    argc = parse_common_args(argc, argv);
    for (i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-ntags") == 0) {
            ntags = atoi(arg_value(argc, argv, &i));
        } else if (strcmp(argv[i], "-sendorder") == 0) {
            sendorder = arg_value(argc, argv, &i);
        } else if (strcmp(argv[i], "-recvorder") == 0) {
            recvorder = arg_value(argc, argv, &i);
        } else if (strcmp(argv[i], "-prepost") == 0) {
            prepost = true;
        } else if (strcmp(argv[i], "-wildcard") == 0) {
            wildcard = true;
        } else {
            if (my_rank == master_rank) {
                fprintf(stderr, "Unknown argument: %s\n", argv[i]);
                MPI_Abort(MPI_COMM_WORLD, -1);
            }
        }
    }

    if (w_size != 2) {
        if (my_rank == master_rank)
            fprintf(stderr, "tagmatch_nb requires exactly 2 ranks, got %d\n", w_size);
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    if (ntags < 1) {
        if (my_rank == master_rank)
            fprintf(stderr, "tagmatch_nb: -ntags must be >= 1, got %d\n", ntags);
        MPI_Abort(MPI_COMM_WORLD, -1);
    }
    /* Sane upper bound: also prevents size_t multiplication overflow on
     * 32-bit hosts where size_t is 32 bits.  2M tags x 1KB msg = 2 GB/rank,
     * already past anything practically useful.                              */
    if (ntags > (1 << 21)) {
        if (my_rank == master_rank)
            fprintf(stderr, "tagmatch_nb: -ntags %d exceeds sane upper bound (%d)\n",
                    ntags, 1 << 21);
        MPI_Abort(MPI_COMM_WORLD, -1);
    }
    /* Cap -ntags at MPI_TAG_UB+1 so every tag we use (0..ntags-1) is legal.
     * MPI_TAG_UB is often INT_MAX; guard against the +1 overflow.            */
    {
        void *attr; int flag;
        MPI_Comm_get_attr(MPI_COMM_WORLD, MPI_TAG_UB, &attr, &flag);
        int tag_ub = (flag && attr) ? *(int*)attr : 32767;
        if (tag_ub < INT_MAX && ntags > tag_ub + 1) {
            if (my_rank == master_rank)
                fprintf(stderr, "tagmatch_nb: -ntags %d exceeds MPI_TAG_UB+1 (%d); capping\n",
                        ntags, tag_ub + 1);
            ntags = tag_ub + 1;
        }
    }

    /* Tag-order arrays (small, plain malloc is fine). */
    int *send_tags = (int*)malloc((size_t)ntags * sizeof(int));
    int *recv_tags = (int*)malloc((size_t)ntags * sizeof(int));
    if (!send_tags || !recv_tags) {
        if (my_rank == master_rank)
            fprintf(stderr, "tagmatch_nb: failed to allocate %d-tag order arrays\n", ntags);
        MPI_Abort(MPI_COMM_WORLD, -1);
    }
    build_order(send_tags, ntags, sendorder, "-sendorder");
    build_order(recv_tags, ntags, recvorder, "-recvorder");

    /* Message buffers + request slots.  Sized ntags * msg_size on each side. */
    const size_t one_buf   = (size_t)msg_size;
    const size_t total_buf = one_buf * (size_t)ntags;
    unsigned char *send_buf = NULL, *recv_buf = NULL;
    unsigned char *send_dev = NULL, *recv_dev = NULL;   /*GPU mode: device mirrors (host pointers in host mode)*/
    MPI_Request *reqs = (MPI_Request*)malloc((size_t)ntags * sizeof(MPI_Request));
    if (!reqs) {
        if (my_rank == master_rank)
            fprintf(stderr, "tagmatch_nb: failed to allocate %d MPI_Requests\n", ntags);
        MPI_Abort(MPI_COMM_WORLD, -1);
    }
    durations = (double*)malloc_align(sizeof(double) * max_samples);

    if (my_rank == 0) {
        send_buf = (unsigned char*)malloc_align(total_buf);
        /* Embed each message's tag in the first sizeof(int) bytes so the
         * receiver can verify the right payload landed in the right slot.
         * Skipped when msg_size < sizeof(int) (degenerate; no integrity check). */
        if (msg_size >= (int)sizeof(int)) {
            for (i = 0; i < ntags; i++) {
                int t = send_tags[i];
                memcpy(send_buf + (size_t)i * one_buf, &t, sizeof(int));
            }
        }
        /*mirror after the host tag write (equivalent to data_push; tags are never rewritten)*/
        send_dev = (unsigned char*)data_mirror(send_buf, total_buf);
    } else {
        recv_buf = (unsigned char*)malloc_align(total_buf);
        recv_dev = (unsigned char*)data_mirror(recv_buf, total_buf);
    }

    if (my_rank == master_rank) {
        if (endless) {
            printf("tagmatch_nb ntags=%d sendorder=%s recvorder=%s prepost=%s wildcard=%s "
                   "msg_size=%d iter=endless\n",
                   ntags, sendorder, recvorder, prepost ? "on" : "off",
                   wildcard ? "on" : "off", msg_size);
        } else {
            printf("tagmatch_nb ntags=%d sendorder=%s recvorder=%s prepost=%s wildcard=%s "
                   "msg_size=%d iter=%d\n",
                   ntags, sendorder, recvorder, prepost ? "on" : "off",
                   wildcard ? "on" : "off", msg_size, max_iters);
        }
    }

    curr_iters     = 0;
    measured_iters = 0;

    double measure_start_time;

    MPI_Barrier(MPI_COMM_WORLD);
    do {
        for (k = 0; k < max_iters + warm_up_iters; k++) {
            if (check_shutdown()) goto done;

            MPI_Barrier(MPI_COMM_WORLD);
            measure_start_time = MPI_Wtime();

            if (my_rank == 0) {
                /* Sender: post all Isends in sendorder, then Waitall. */
                for (i = 0; i < ntags; i++) {
                    int t = send_tags[i];
                    MPI_Isend(send_dev + (size_t)i * one_buf, msg_size, MPI_BYTE,
                              1, t, MPI_COMM_WORLD, &reqs[i]);
                }
                MPI_Waitall(ntags, reqs, MPI_STATUSES_IGNORE);
            } else {
                /* Receiver: two scheduling patterns. */
                if (prepost) {
                    /* Targets PRQ path: post all N Irecvs first, then Waitall. */
                    for (i = 0; i < ntags; i++) {
                        int t = wildcard ? MPI_ANY_TAG : recv_tags[i];
                        MPI_Irecv(recv_dev + (size_t)i * one_buf, msg_size, MPI_BYTE,
                                  0, t, MPI_COMM_WORLD, &reqs[i]);
                    }
                    MPI_Waitall(ntags, reqs, MPI_STATUSES_IGNORE);
                } else {
                    /* Targets UMQ path: Irecv-then-Wait per message; while
                     * the receiver waits, other messages may accumulate
                     * in the UMQ. */
                    for (i = 0; i < ntags; i++) {
                        int t = wildcard ? MPI_ANY_TAG : recv_tags[i];
                        MPI_Request req;
                        MPI_Irecv(recv_dev + (size_t)i * one_buf, msg_size, MPI_BYTE,
                                  0, t, MPI_COMM_WORLD, &req);
                        MPI_Wait(&req, MPI_STATUS_IGNORE);
                    }
                }
            }

            if (k >= warm_up_iters) record_duration(MPI_Wtime() - measure_start_time);
            curr_iters++;
        }
    } while (endless);

done:
    if (debug_mode) {
        int check_ok = 1;
        /* Receiver verifies: with non-wildcard, slot i was posted for tag
         * recv_tags[i] and must therefore contain the sender's payload for
         * that tag — which encodes the tag itself in its first 4 bytes.
         * Wildcard mode is order-indeterminate, so we skip verification. */
        if (my_rank == 1 && !wildcard && msg_size >= (int)sizeof(int)) {
            data_fetch(recv_buf, recv_dev, total_buf);
            for (i = 0; i < ntags && check_ok; i++) {
                int got;
                memcpy(&got, recv_buf + (size_t)i * one_buf, sizeof(int));
                if (got != recv_tags[i]) check_ok = 0;
            }
        }
        printf("DEBUG rank=%d nprocs=%d ntags=%d sendorder=%s recvorder=%s "
               "prepost=%d wildcard=%d check=%s\n",
               my_rank, w_size, ntags, sendorder, recvorder,
               prepost ? 1 : 0, wildcard ? 1 : 0, check_ok ? "OK" : "FAIL");
        fflush(stdout);
        MPI_Barrier(MPI_COMM_WORLD);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    write_results();

    free(reqs);
    free(send_tags);
    free(recv_tags);
    if (send_buf) free(send_buf);
    if (recv_buf) free(recv_buf);
    if (send_dev) data_free(send_dev);
    if (recv_dev) data_free(recv_dev);
    free(durations);

    MPI_Finalize();
    return 0;
}
