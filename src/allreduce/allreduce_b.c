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
    int send_buf_size, recv_buf_size;
    int *send_buf;
    int *recv_buf;
    
    if(msg_size%sizeof(int)!=0){
        if(my_rank==master_rank)
            fprintf(stderr, "Msg-size (%d) must be divisible by size of int (%zu)\n",msg_size,sizeof(int));
        MPI_Abort(MPI_COMM_WORLD, -1);
    }
    
    msg_size_ints=msg_size/sizeof(int);
    send_buf_size=msg_size;
    recv_buf_size=msg_size;
    
    send_buf=(int*)malloc_align(send_buf_size);
    recv_buf=(int*)malloc_align(recv_buf_size);
    durations=(double *)malloc_align(sizeof(double)*max_samples);
    
    
    /*fill send buffer with dummies*/
    for(i=0;i<msg_size_ints;i++){
        send_buf[i] = debug_mode ? my_rank : 1;
    }

    /*GPU mode: MPI works on device mirrors; in host mode the mirrors are the host buffers*/
    int *send_dev = (int*)data_mirror(send_buf, msg_size);
    int *recv_dev = (int*)data_mirror(recv_buf, msg_size);

    
    /*print basic info to stdout*/
    if(my_rank==master_rank){
        if(endless){
            printf("All-reduce with %d processes, msg-size: %d, test iterations: endless.\n"
                    ,w_size,msg_size);
        }else{
            printf("All-reduce with %d processes, msg-size: %d, test iterations: %d.\n"
                    ,w_size,msg_size,max_iters);
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
                    MPI_Allreduce(send_dev,recv_dev,msg_size_ints,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
                }
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
        data_fetch(recv_buf, recv_dev, msg_size);
        int _e = w_size * (w_size - 1) / 2, _ok = 1, _b;
        for (_b = 0; _b < msg_size_ints && _ok; _b++)
            if (recv_buf[_b] != _e) _ok = 0;
        printf("DEBUG rank=%d nprocs=%d check=%s\n", my_rank, w_size, _ok ? "OK" : "FAIL");
        fflush(stdout);
        MPI_Barrier(MPI_COMM_WORLD);
    }
done:
    /*write results to file*/
    MPI_Barrier(MPI_COMM_WORLD);
    write_results();
    
    /*free allocated buffers*/
    free(durations);
    free(recv_buf);
    free(send_buf);
    data_free(send_dev);
    data_free(recv_dev);
    
    /*exit MPI library*/
    MPI_Finalize();
}

