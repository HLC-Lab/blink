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

/* Nonblocking reduction: -gpu additionally probes MPI_Iallreduce/Ireduce/Ireduce_scatter on device memory. */
int benchmark_uses_nb_reduce = 1;

int main(int argc, char** argv){

    /*init MPI world*/
    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD, &w_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    /*register signal handler*/
    install_shutdown_handler();

    /*parse command line*/
    int i, k;
    argc = parse_common_args(argc, argv);
    for (i = 1; i < argc; i++) {
        if (my_rank == master_rank) {
            fprintf(stderr, "Unknown argument: %s\n", argv[i]);
            MPI_Abort(MPI_COMM_WORLD, -1);
        }
    }

    /*pin to core*/
    /*cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(1, &mask);
    sched_setaffinity(0, sizeof(mask), &mask);*/

    /*allocate buffers*/
    int msg_size_ints;
    int *send_buf;
    int *recv_buf;
    MPI_Request *requests;

    if(msg_size%sizeof(int)!=0){
        if(my_rank==master_rank)
            fprintf(stderr, "Msg-size (%d) must be divisible by size of int (%zu)\n",msg_size,sizeof(int));
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    msg_size_ints=msg_size/sizeof(int);

    send_buf=(int*)malloc_align(msg_size);
    /* one reduced result per batched (granularity) call so the concurrently
     * outstanding MPI_Ireduce ops never share a receive buffer (-grty > 1). */
    recv_buf=(int*)malloc_align((size_t)measure_granularity*msg_size);
    durations=(double *)malloc_align(sizeof(double)*max_samples);
    requests=(MPI_Request*)malloc_align(sizeof(MPI_Request)*measure_granularity);


    /*fill send buffer with dummies*/
    for(i=0;i<msg_size_ints;i++){
        send_buf[i] = debug_mode ? my_rank : 1;
    }

    /*GPU mode: MPI works on device mirrors; in host mode the mirrors are the host buffers*/
    int *send_dev = (int*)data_mirror(send_buf, msg_size);
    int *recv_dev = (int*)data_mirror(recv_buf, (size_t)measure_granularity*msg_size);

    /*print basic info to stdout*/
    if(my_rank==master_rank){
        if(endless){
            printf("Reduce with %d processes, root: %d, msg-size: %d, test iterations: endless.\n"
                    ,w_size,master_rank,msg_size);
        }else{
            printf("Reduce with %d processes, root: %d, msg-size: %d, test iterations: %d.\n"
                    ,w_size,master_rank,msg_size,max_iters);
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
                measure_start_time=MPI_Wtime();
                for(i=0;i<measure_granularity;i++){
                    MPI_Ireduce(send_dev,&recv_dev[(size_t)i*msg_size_ints],msg_size_ints,MPI_INT,MPI_SUM,master_rank,MPI_COMM_WORLD,&requests[i]);
                }
                MPI_Waitall(measure_granularity,requests,MPI_STATUSES_IGNORE);
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
        data_fetch(recv_buf, recv_dev, (size_t)measure_granularity*msg_size);
        int _e = w_size * (w_size - 1) / 2, _ok = 1, _b;
        if (my_rank == master_rank)
            for (_b = 0; _b < msg_size_ints && _ok; _b++)
                if (recv_buf[_b] != _e) _ok = 0;
        printf("DEBUG rank=%d nprocs=%d root=%d check=%s\n",
               my_rank, w_size, master_rank, _ok ? "OK" : "FAIL");
        fflush(stdout);
        MPI_Barrier(MPI_COMM_WORLD);
    }
done:
    /*write results to file*/
    MPI_Barrier(MPI_COMM_WORLD);
    write_results();

    /*free allocated buffers*/
    free(durations);
    free(send_buf);
    free(recv_buf);
    data_free(send_dev);
    data_free(recv_dev);
    free(requests);

    /*exit MPI library*/
    MPI_Finalize();
}
