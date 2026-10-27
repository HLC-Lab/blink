#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <signal.h>
#include <stdbool.h>
#include <sched.h>
#include "common.h"

const char *benchmark_help =
"  -k <int>                      number of random partners per rank (default 1, capped at w_size-1)\n";

int main(int argc, char** argv){

    /*init MPI world*/
    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD, &w_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    /*register signal handler*/
    install_shutdown_handler();

    int npartners=1; /*number of random communication partners per rank (-k flag)*/

    /*parse command line*/
    int i, k, p;
    argc = parse_common_args(argc, argv);
    for (i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-k") == 0) {
            npartners = atoi(arg_value(argc, argv, &i));
        } else {
            if (my_rank == master_rank) {
                fprintf(stderr, "Unknown argument: %s\n", argv[i]);
                MPI_Abort(MPI_COMM_WORLD, -1);
            }
        }
    }

    /*validate npartners*/
    if(npartners<1){
        if(my_rank==master_rank){
            fprintf(stderr,"k (%d) must be >= 1\n",npartners);
            MPI_Abort(MPI_COMM_WORLD, -1);
        }
    }
    if(npartners>w_size-1){
        if(my_rank==master_rank){
            fprintf(stderr,"Warning: k (%d) capped to w_size-1 (%d)\n",npartners,w_size-1);
        }
        npartners=w_size-1;
    }

    /*
     * Build npartners random permutations of [0, w_size-1] such that the
     * resulting communication graph is k-regular AND every rank has k
     * DISTINCT partners (no self-pairings, no duplicate target across
     * permutations).  For permutation p, rank i sends to perms[p*w_size + i].
     *
     * Algorithm: rejection sampling.  Each permutation must
     *   (a) have no fixed point (perms[p][i] != i — avoids self-send), and
     *   (b) for every i, perms[p][i] != perms[q][i] for all q < p (avoids
     *       column duplicates, i.e. same target as an earlier permutation).
     * If a generated permutation fails, retry.  For small k relative to
     * w_size this terminates quickly.  After many attempts fail we fall
     * back to a swap-repair pass.  Partners are fixed for the entire run.
     */
    int *perms;
    perms=(int*)malloc_align(sizeof(int)*npartners*w_size);
    if(perms==NULL){
        fprintf(stderr,"Failed to allocate permutation table on rank %d\n",my_rank);
        MPI_Abort(MPI_COMM_WORLD, -1);
    }
    int attempts;
    for(p=0;p<npartners;p++){
        int valid = 0;
        for(attempts=0; attempts<1000 && !valid; attempts++){
            for(i=0;i<w_size;i++) perms[p*w_size+i]=i;
            permute(&perms[p*w_size],w_size);
            valid = 1;
            for(i=0;i<w_size && valid;i++){
                int t = perms[p*w_size+i];
                if (t == i) { valid = 0; break; }   /* (a) no self-pairing  */
                int q;
                for(q=0;q<p;q++){
                    if (perms[q*w_size+i] == t) { valid = 0; break; }
                }
            }
        }
        if (!valid) {
            /* swap-repair fallback: walk through and swap any conflict with
             * a later index in the same row.                                */
            for(i=0;i<w_size;i++){
                int t = perms[p*w_size+i];
                int needs_swap = (t == i);
                if (!needs_swap) {
                    int q;
                    for(q=0;q<p && !needs_swap;q++)
                        if (perms[q*w_size+i] == t) needs_swap = 1;
                }
                if (needs_swap) {
                    int j;
                    for(j=i+1;j<w_size;j++){
                        int tj = perms[p*w_size+j];
                        int ok = (tj != i) && (t != j);
                        int q;
                        for(q=0;q<p && ok;q++){
                            if (perms[q*w_size+i] == tj) ok = 0;
                            if (perms[q*w_size+j] == t)  ok = 0;
                        }
                        if (ok) {
                            perms[p*w_size+i] = tj;
                            perms[p*w_size+j] = t;
                            break;
                        }
                    }
                }
            }
        }
    }

    /*pin to core*/
    /*cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(1, &mask);
    sched_setaffinity(0, sizeof(mask), &mask);*/

    /*allocate buffers*/
    size_t send_buf_size, recv_buf_size;
    unsigned char *send_buf;
    unsigned char *recv_buf;
    MPI_Request *send_requests;
    MPI_Request *recv_requests;

    send_buf_size=msg_size;
    recv_buf_size=(size_t)npartners*measure_granularity*msg_size;

    send_buf=(unsigned char*)malloc_align(send_buf_size);
    recv_buf=(unsigned char*)malloc_align(recv_buf_size);
    durations=(double *)malloc_align(sizeof(double)*max_samples);
    send_requests=(MPI_Request*)malloc_align(sizeof(MPI_Request)*npartners*measure_granularity);
    recv_requests=(MPI_Request*)malloc_align(sizeof(MPI_Request)*npartners*measure_granularity);


    /*fill send buffer with dummies*/
    for(i=0;i<send_buf_size;i++){
        send_buf[i] = debug_mode ? (unsigned char)my_rank : 'a';
    }

    /*GPU mode: MPI works on device mirrors; in host mode the mirrors are the host buffers*/
    unsigned char *send_dev = (unsigned char*)data_mirror(send_buf, send_buf_size);
    unsigned char *recv_dev = (unsigned char*)data_mirror(recv_buf, recv_buf_size);

    /*print basic info to stdout*/
    if(my_rank==master_rank){
        if(endless){
            printf("k-random-partners with %d processes, k=%d, msg-size: %d, test iterations: endless.\n"
                    ,w_size,npartners,msg_size);
        }else{
            printf("k-random-partners with %d processes, k=%d, msg-size: %d, test iterations: %d.\n"
                    ,w_size,npartners,msg_size,max_iters);
        }
    }

    /*measured iterations*/
    double burst_start_time;
    double measure_start_time;
    double burst_length_mean=burst_length;
    double burst_pause_mean=burst_pause;
    int burst_cont=0;
    curr_iters=0;
    measured_iters=0;

    int antideadlock_tag=0;

    MPI_Barrier(MPI_COMM_WORLD);
    do{
        for(k=0;k<max_iters+warm_up_iters;k++){
            if (check_shutdown()) goto done;
            if(burst_length_rand){ /*randomized burst length*/
                burst_length=sample_burst_length(burst_length_mean);
            }
            burst_start_time=MPI_Wtime();
            do{
                MPI_Barrier(MPI_COMM_WORLD);
                antideadlock_tag=0; /* reset per timed window so the tag never grows unbounded (stays < MPI_TAG_UB on long/endless runs) */
                measure_start_time=MPI_Wtime();
                for(i=0;i<measure_granularity;i++){
                    /*post npartners receives (MPI_ANY_SOURCE: each perm maps exactly one sender to this rank)*/
                    /*post npartners sends to this rank's target in each permutation*/
                    for(p=0;p<npartners;p++){
                        MPI_Irecv(&recv_dev[(i*npartners+p)*msg_size],msg_size,MPI_BYTE,
                                  MPI_ANY_SOURCE,antideadlock_tag,MPI_COMM_WORLD,
                                  &recv_requests[i*npartners+p]);
                        MPI_Isend(send_dev,msg_size,MPI_BYTE,
                                  perms[p*w_size+my_rank],antideadlock_tag,MPI_COMM_WORLD,
                                  &send_requests[i*npartners+p]);
                    }
                    antideadlock_tag++;
                }
                MPI_Waitall(npartners*measure_granularity,send_requests,MPI_STATUSES_IGNORE);
                MPI_Waitall(npartners*measure_granularity,recv_requests,MPI_STATUSES_IGNORE);
                if (k >= warm_up_iters) record_duration(MPI_Wtime()-measure_start_time);
                curr_iters++;
                if(burst_length!=0){ /*bcast needed for synch if bursts timed*/
                    if(my_rank==master_rank){ /*master decides if burst should be continued*/
                        burst_cont=((MPI_Wtime()-burst_start_time)<burst_length);
                    }
                    MPI_Bcast(&burst_cont,1,MPI_INT,master_rank,MPI_COMM_WORLD); /*bcast the masters decision*/
                }
            }while(burst_cont);
            if(burst_pause!=0){
                if(burst_pause_rand){ /*randomized break length*/
                    burst_pause=sample_pause_length(burst_pause_mean);
                }
                dsleep(burst_pause);
            }
        }
    }while(endless);

    if (debug_mode) {
        /* Deterministic post-loop exchange: for each permutation p, invert
         * it locally to discover which rank sends to me (since rejection
         * sampling guarantees distinct senders per row, this is unique).
         * Then receive from that specific source and verify the payload is
         * the sender's rank value.  Barrier first to drain any
         * measurement-loop ANY_TAG receives so they don't consume our
         * debug-tagged sends. */
        MPI_Barrier(MPI_COMM_WORLD);
        int _p, _b, _ok = 1;
        int *inverse = (int*)malloc_align(sizeof(int)*npartners*w_size);
        if (inverse == NULL) {
            fprintf(stderr,"Failed to allocate inverse perms on rank %d\n",my_rank);
            MPI_Abort(MPI_COMM_WORLD, -1);
        }
        for (_p = 0; _p < npartners; _p++) {
            int q;
            for (q = 0; q < w_size; q++) inverse[_p*w_size + perms[_p*w_size + q]] = q;
        }
        MPI_Request *dbg_reqs = (MPI_Request*)malloc_align(sizeof(MPI_Request)*npartners*2);
        for (_p = 0; _p < npartners; _p++) {
            int from = inverse[_p*w_size + my_rank];
            int to   = perms[_p*w_size + my_rank];
            MPI_Irecv(&recv_dev[_p*msg_size], msg_size, MPI_BYTE, from, 0xD0+_p,
                      MPI_COMM_WORLD, &dbg_reqs[_p]);
            MPI_Isend(send_dev, msg_size, MPI_BYTE, to, 0xD0+_p,
                      MPI_COMM_WORLD, &dbg_reqs[npartners + _p]);
        }
        MPI_Waitall(npartners*2, dbg_reqs, MPI_STATUSES_IGNORE);
        data_fetch(recv_buf, recv_dev, recv_buf_size);
        for (_p = 0; _p < npartners && _ok; _p++) {
            int expected_from = inverse[_p*w_size + my_rank];
            for (_b = 0; _b < msg_size && _ok; _b++)
                if (recv_buf[(size_t)_p*msg_size + _b] != (unsigned char)expected_from) _ok = 0;
        }
        free(dbg_reqs);
        free(inverse);
        printf("DEBUG rank=%d nprocs=%d k=%d check=%s\n",
               my_rank, w_size, npartners, _ok ? "OK" : "FAIL");
        fflush(stdout);
        MPI_Barrier(MPI_COMM_WORLD);
    }
done:
    /*write results to file*/
    MPI_Barrier(MPI_COMM_WORLD);
    write_results();

    /*free allocated buffers*/
    free(perms);
    free(durations);
    free(send_buf);
    free(recv_buf);
    data_free(send_dev);
    data_free(recv_dev);
    free(send_requests);
    free(recv_requests);

    /*exit MPI library*/
    MPI_Finalize();
}
