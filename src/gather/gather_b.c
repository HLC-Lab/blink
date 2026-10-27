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
    /*all ranks need a send buffer; only root needs recv buffer large enough for all ranks*/
    unsigned char *send_buf;
    unsigned char *recv_buf=NULL;

    send_buf=(unsigned char*)malloc_align(msg_size);
    durations=(double *)malloc_align(sizeof(double)*max_samples);

    if(my_rank==master_rank){
        recv_buf=(unsigned char*)malloc_align((size_t)w_size*msg_size);
        if(recv_buf==NULL){
            fprintf(stderr,"Failed to allocate recv_buf on rank %d\n",my_rank);
            MPI_Abort(MPI_COMM_WORLD, -1);
        }
    }


    /*fill send buffer with dummies*/
    for(i=0;i<msg_size;i++){
        send_buf[i] = debug_mode ? (unsigned char)my_rank : 'a';
    }

    /*GPU mode: MPI works on device mirrors; in host mode the mirrors are the host buffers*/
    unsigned char *send_dev = (unsigned char*)data_mirror(send_buf, msg_size);
    unsigned char *recv_dev = NULL;   /*only the root has a receive buffer*/
    if(my_rank==master_rank) recv_dev = (unsigned char*)data_mirror(recv_buf, (size_t)w_size*msg_size);

    /*print basic info to stdout*/
    if(my_rank==master_rank){
        if(endless){
            printf("Gather with %d processes, root: %d, msg-size: %d, test iterations: endless.\n"
                    ,w_size,master_rank,msg_size);
        }else{
            printf("Gather with %d processes, root: %d, msg-size: %d, test iterations: %d.\n"
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
                    MPI_Gather(send_dev,msg_size,MPI_BYTE,recv_dev,msg_size,MPI_BYTE,master_rank,MPI_COMM_WORLD);
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
        if (my_rank == master_rank) data_fetch(recv_buf, recv_dev, (size_t)w_size*msg_size);
        int _r, _b, _ok = 1;
        if (my_rank == master_rank)
            for (_r = 0; _r < w_size && _ok; _r++)
                for (_b = 0; _b < msg_size && _ok; _b++)
                    if (recv_buf[(size_t)_r * msg_size + _b] != (unsigned char)_r) _ok = 0;
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
    if(my_rank==master_rank) free(recv_buf);
    data_free(send_dev);
    if(my_rank==master_rank) data_free(recv_dev);

    /*exit MPI library*/
    MPI_Finalize();
}
