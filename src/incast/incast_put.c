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

/* One-sided benchmark: -gpu additionally probes MPI_Win/Put/Get on device memory. */
int benchmark_uses_rma = 1;

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
    size_t send_buf_size, recv_buf_size;
    unsigned char *send_buf;
    unsigned char *recv_buf;
    MPI_Win rma_win;
    
    send_buf_size=msg_size;
    recv_buf_size=(size_t)w_size*msg_size*measure_granularity;
    
    send_buf=(unsigned char*)malloc_align(send_buf_size);
    recv_buf=(unsigned char*)malloc_align(recv_buf_size);
    durations=(double *)malloc_align(sizeof(double)*max_samples);
    
    /*fill send buffer: rank-as-payload under -debug so the master can verify
     * delivered RMA contents.                                              */
    if(my_rank!=master_rank){
        for(size_t bi=0;bi<send_buf_size;bi++){
            send_buf[bi] = debug_mode ? (unsigned char)my_rank : 'a';
        }
    }

    /*GPU mode: MPI works on device mirrors; in host mode the mirrors are the host buffers*/
    unsigned char *send_dev = (unsigned char*)data_mirror(send_buf, send_buf_size);
    unsigned char *recv_dev = (unsigned char*)data_mirror(recv_buf, recv_buf_size);

    /*window over the device mirror (in host mode the host buffer)*/
    MPI_Win_create(recv_dev, recv_buf_size, 1, MPI_INFO_NULL, MPI_COMM_WORLD, &rma_win);

    /*print basic info to stdout*/
    if(my_rank==master_rank){
        if(endless){
            printf("Incast with %d processes, receiver rank: %d, msg-size: %d, test iterations: endless.\n"
                    ,w_size,master_rank,msg_size);
        }else{
            printf("Incast with %d processes, receiver rank: %d, msg-size: %d, test iterations: %d.\n"
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
                MPI_Win_fence(MPI_MODE_NOPRECEDE,rma_win);
                for(i=0;i<measure_granularity;i++){
                    if(my_rank!=master_rank){
                        MPI_Put(send_dev,msg_size,MPI_BYTE,master_rank
                            ,(MPI_Aint)my_rank*msg_size*measure_granularity+(MPI_Aint)i*msg_size,msg_size,MPI_BYTE,rma_win);
                    }
                }
                MPI_Win_fence(MPI_MODE_NOSUCCEED,rma_win);
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

    /* debug: master verifies each sender's Put-payload landed at the right
     * offset in its window.  Senders contribute nothing here beyond having
     * filled send_buf with their rank.                                    */
    if (debug_mode) {
        if (my_rank == master_rank) {
            data_fetch(recv_buf, recv_dev, recv_buf_size);
            int s, _b, _ok = 1;
            for (s = 0; s < w_size && _ok; s++) {
                if (s == master_rank) continue;
                for (_b = 0; _b < msg_size && _ok; _b++)
                    if (recv_buf[(size_t)s*msg_size*measure_granularity + _b] != (unsigned char)s) _ok = 0;
            }
            printf("DEBUG rank=%d nprocs=%d nsenders=%d check=%s\n",
                   my_rank, w_size, w_size - 1, _ok ? "OK" : "FAIL");
        } else {
            printf("DEBUG rank=%d nprocs=%d target=%d check=OK\n",
                   my_rank, w_size, master_rank);
        }
        fflush(stdout);
        MPI_Barrier(MPI_COMM_WORLD);
    }
done:
    /*write results to file*/
    MPI_Barrier(MPI_COMM_WORLD);
    write_results();
    
    /*free allocated buffers*/
    MPI_Win_free(&rma_win);
    free(durations);
    free(recv_buf);
    free(send_buf);
    data_free(send_dev);
    data_free(recv_dev);
    
    /*exit MPI library*/
    MPI_Finalize();
}

