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
    int i, j, k;
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
    
    send_buf_size=msg_size;
    recv_buf_size=(size_t)(w_size-1)*msg_size;
    
    send_buf=(unsigned char*)malloc_align(send_buf_size);
    recv_buf=(unsigned char*)malloc_align(recv_buf_size);
    durations=(double *)malloc_align(sizeof(double)*max_samples);
    
    
    /*fill send buffer: rank-as-payload under -debug so receivers can verify*/
    if(my_rank!=master_rank){
        for(size_t bi=0;bi<send_buf_size;bi++){
            send_buf[bi] = debug_mode ? (unsigned char)my_rank : 'a';
        }
    }

    /*GPU mode: MPI works on device mirrors; in host mode the mirrors are the host buffers*/
    unsigned char *send_dev = (unsigned char*)data_mirror(send_buf, send_buf_size);
    unsigned char *recv_dev = (unsigned char*)data_mirror(recv_buf, recv_buf_size);

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
                for(i=0;i<measure_granularity;i++){
                    if (my_rank==master_rank){
                        for(j=0;j<w_size-1;j++){
                            MPI_Recv(&recv_dev[j*msg_size],msg_size,MPI_BYTE, MPI_ANY_SOURCE
                                    ,MPI_ANY_TAG, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
                        }
                    }else{
                        MPI_Send(send_dev,msg_size,MPI_BYTE,master_rank,my_rank,MPI_COMM_WORLD);
                    }
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

    /* debug: deterministic post-loop exchange that verifies each sender's
     * payload arrived intact at the receiver.  Uses specific source/tag so
     * we can attribute each slot to a known sender.  Barrier first to
     * drain any measurement-loop ANY_TAG receives so they don't consume
     * our debug-tagged sends.                                              */
    if (debug_mode) {
        MPI_Barrier(MPI_COMM_WORLD);
        int _ok = 1;
        if (my_rank == master_rank) {
            int s, _b, slot = 0;
            for (s = 0; s < w_size; s++) {
                if (s == master_rank) continue;
                MPI_Recv(&recv_dev[slot*msg_size], msg_size, MPI_BYTE,
                         s, 0xD, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                data_fetch(&recv_buf[slot*msg_size], &recv_dev[slot*msg_size], msg_size);
                for (_b = 0; _b < msg_size && _ok; _b++)
                    if (recv_buf[(size_t)slot*msg_size + _b] != (unsigned char)s) _ok = 0;
                slot++;
            }
            printf("DEBUG rank=%d nprocs=%d nsenders=%d check=%s\n",
                   my_rank, w_size, w_size - 1, _ok ? "OK" : "FAIL");
        } else {
            MPI_Send(send_dev, msg_size, MPI_BYTE, master_rank, 0xD, MPI_COMM_WORLD);
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
    free(durations);
    free(recv_buf);
    free(send_buf);
    data_free(send_dev);
    data_free(recv_dev);
    
    /*exit MPI library*/
    MPI_Finalize();
}

