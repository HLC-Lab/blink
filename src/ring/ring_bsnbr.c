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
"  -rring                        randomise the ring topology (default: sequential 0->1->...->N-1)\n";

int main(int argc, char** argv){

    /*init MPI world*/
    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD, &w_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    
    /*register signal handler*/
    install_shutdown_handler();

    bool rand_ring=false;

    /*parse command line*/
    int i, k;
    argc = parse_common_args(argc, argv);
    for (i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-rring") == 0) {
            rand_ring = true;
        } else {
            if (my_rank == master_rank) {
                fprintf(stderr, "Unknown argument: %s\n", argv[i]);
                MPI_Abort(MPI_COMM_WORLD, -1);
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
    int *targets;
    MPI_Request *recv_requests;

    send_buf_size=msg_size;
    recv_buf_size=(size_t)2*measure_granularity*msg_size;
    
    send_buf=(unsigned char*)malloc_align(send_buf_size);
    recv_buf=(unsigned char*)malloc_align(recv_buf_size);
    targets=(int*)malloc_align(sizeof(int)*w_size);
    durations=(double *)malloc_align(sizeof(double)*max_samples);
    recv_requests=(MPI_Request*)malloc_align(sizeof(MPI_Request)*2*measure_granularity);
    
    
    /*fill send buffer: rank-as-payload under -debug so neighbours can verify*/
    for(i=0;i<send_buf_size;i++){
        send_buf[i] = debug_mode ? (unsigned char)my_rank : 'a';
    }

    /*GPU mode: MPI works on device mirrors; in host mode the mirrors are the host buffers*/
    unsigned char *send_dev = (unsigned char*)data_mirror(send_buf, send_buf_size);
    unsigned char *recv_dev = (unsigned char*)data_mirror(recv_buf, recv_buf_size);
    
    /*setup ring*/
    for(i=0;i<w_size;i++){
        targets[i]=i;
    }
    if(rand_ring){
        permute(targets,w_size);
    }

    /* Treat targets[] as "position -> rank" map; build the inverse so each
     * rank can find its position in the (possibly permuted) ring.  Then the
     * left/right neighbours are the ranks at adjacent positions, which
     * guarantees a single Hamiltonian cycle in both directions and full
     * reciprocity (A.left = B  <=>  B.right = A).                          */
    int *positions = (int*)malloc_align(sizeof(int)*w_size);
    if (positions == NULL) {
        fprintf(stderr, "Failed to allocate positions on rank %d\n", my_rank);
        MPI_Abort(MPI_COMM_WORLD, -1);
    }
    for(i=0;i<w_size;i++) positions[targets[i]] = i;
    int my_pos = positions[my_rank];
    int left_neighbor  = targets[mod(my_pos - 1, w_size)];
    int right_neighbor = targets[mod(my_pos + 1, w_size)];
    int antideadlock_tag;

    /*print basic info to stdout*/
    if(my_rank==master_rank){
        if(endless){
            printf("Ring with %d processes, randomized: %s, msg-size: %d, test iterations: endless.\n"
                    ,w_size,(rand_ring?"true":"false"),msg_size);
        }else{
            printf("Ring with %d processes, randomized: %s, msg-size: %d, test iterations: %d.\n"
                    ,w_size,(rand_ring?"true":"false"),msg_size,max_iters);
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
    
    antideadlock_tag=0;
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
                    MPI_Irecv(&recv_dev[2*i*msg_size],msg_size,MPI_BYTE,MPI_ANY_SOURCE
                            ,antideadlock_tag, MPI_COMM_WORLD,&recv_requests[2*i]);
                    MPI_Irecv(&recv_dev[(2*i+1)*msg_size],msg_size,MPI_BYTE,MPI_ANY_SOURCE
                            ,antideadlock_tag, MPI_COMM_WORLD,&recv_requests[2*i+1]);
                    MPI_Send(send_dev,msg_size,MPI_BYTE,left_neighbor,antideadlock_tag,MPI_COMM_WORLD);
                    MPI_Send(send_dev,msg_size,MPI_BYTE,right_neighbor,antideadlock_tag,MPI_COMM_WORLD);
                    antideadlock_tag++;
                }
                MPI_Waitall(2*measure_granularity,recv_requests,MPI_STATUSES_IGNORE);
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

    /* debug: deterministic post-loop exchange to verify both ring neighbours
     * sent the expected payload (their rank).  Uses specific sources so a
     * broken topology (e.g. the old -rring bug) is detected.  Barrier
     * first to drain any measurement-loop ANY_TAG receives.                */
    if (debug_mode) {
        MPI_Barrier(MPI_COMM_WORLD);
        /* reuse the already-allocated recv_buf (>= 2*msg_size) instead of a
         * stack VLA, which would risk a stack overflow for large msg_size   */
        unsigned char *left_buf  = recv_buf;
        unsigned char *right_buf = recv_buf + msg_size;
        unsigned char *left_dev  = recv_dev;
        unsigned char *right_dev = recv_dev + msg_size;
        MPI_Request reqs[2];
        int _ok = 1, _b;
        MPI_Irecv(left_dev,  msg_size, MPI_BYTE, left_neighbor,  0xD, MPI_COMM_WORLD, &reqs[0]);
        MPI_Irecv(right_dev, msg_size, MPI_BYTE, right_neighbor, 0xD, MPI_COMM_WORLD, &reqs[1]);
        MPI_Send(send_dev,   msg_size, MPI_BYTE, left_neighbor,  0xD, MPI_COMM_WORLD);
        MPI_Send(send_dev,   msg_size, MPI_BYTE, right_neighbor, 0xD, MPI_COMM_WORLD);
        MPI_Waitall(2, reqs, MPI_STATUSES_IGNORE);
        data_fetch(recv_buf, recv_dev, recv_buf_size);
        for (_b = 0; _b < msg_size && _ok; _b++)
            if (left_buf[_b]  != (unsigned char)left_neighbor)  _ok = 0;
        for (_b = 0; _b < msg_size && _ok; _b++)
            if (right_buf[_b] != (unsigned char)right_neighbor) _ok = 0;
        printf("DEBUG rank=%d nprocs=%d left=%d right=%d check=%s\n",
               my_rank, w_size, left_neighbor, right_neighbor, _ok ? "OK" : "FAIL");
        fflush(stdout);
        MPI_Barrier(MPI_COMM_WORLD);
    }
done:
    /*write results to file*/
    MPI_Barrier(MPI_COMM_WORLD);
    write_results();
    
    /*free allocated buffers*/
    free(targets);
    free(positions);
    free(durations);
    free(send_buf);
    free(recv_buf);
    data_free(send_dev);
    data_free(recv_dev);
    free(recv_requests);
    
    /*exit MPI library*/
    MPI_Finalize();
}

