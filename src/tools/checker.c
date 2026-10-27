#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <signal.h>
#include <stdbool.h>
#include <sys/time.h>
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

    /*allocate buffers*/
    size_t send_buf_size, recv_buf_size;
    unsigned char *send_buf;
    unsigned char *recv_buf;

    send_buf_size=(size_t)msg_size*w_size;
    recv_buf_size=(size_t)msg_size*w_size;

    send_buf=(unsigned char*)malloc_align(send_buf_size);
    recv_buf=(unsigned char*)malloc_align(recv_buf_size);
    durations=(double *)malloc_align(sizeof(double)*max_samples);


    /*fill send buffer with dummies*/
    for(size_t bi=0;bi<send_buf_size;bi++){
        send_buf[bi]='a';
    }

    /*GPU mode: MPI works on device mirrors; in host mode the mirrors are the host buffers*/
    unsigned char *send_dev = (unsigned char*)data_mirror(send_buf, send_buf_size);
    unsigned char *recv_dev = (unsigned char*)data_mirror(recv_buf, recv_buf_size);

    /*print basic info to stdout*/
    if(my_rank==master_rank){
        if(endless){
            printf("All-to-all with %d processes, msg-size: %d, test iterations: endless.\n"
                    ,w_size,msg_size);
        }else{
            printf("All-to-all with %d processes, msg-size: %d, test iterations: %d.\n"
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

    //-----------------
    /* per-iteration timestamp log — opened and written only on the master rank */
    FILE *fd_temp=NULL;
    if(my_rank==master_rank){
        time_t t = time(NULL);
        struct tm *tm = localtime(&t);
        char s[64];
        strftime(s, sizeof(s), "checker_%Y%m%d_%H%M%S.log", tm);
        fd_temp=fopen(s,"w");
        if(fd_temp==NULL){
            fprintf(stderr,"Failed to open log file %s on rank %d\n",s,my_rank);
            MPI_Abort(MPI_COMM_WORLD, -1);
        }
        struct timeval time_now;
        gettimeofday(&time_now, NULL);
        struct tm *time_str_tm;
        time_str_tm = localtime(&time_now.tv_sec); /* local time, consistent with the local-time log filename */
        if(endless){
            fprintf(fd_temp, "endless, %i B, %i iter, %i grty\n",msg_size,max_iters,measure_granularity);
        }else{
            fprintf(fd_temp, "limited, %i B, %i iter, %i grty\n",msg_size,max_iters,measure_granularity);
        }
        fprintf(fd_temp, "%02i:%02i:%02i:%06li\n---------------\n"
           , time_str_tm->tm_hour
           , time_str_tm->tm_min
           , time_str_tm->tm_sec
           , time_now.tv_usec);
        fflush(fd_temp);
    }
    //-----------------

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
                    MPI_Alltoall(send_dev,msg_size,MPI_BYTE,recv_dev,msg_size,MPI_BYTE,MPI_COMM_WORLD);
                }
                if (k >= warm_up_iters) record_duration(MPI_Wtime()-measure_start_time); /*write result to ring buffer*/
                curr_iters++;
                //-----------------
                if(my_rank==master_rank){
                    struct timeval time_now;
                    gettimeofday(&time_now, NULL);
                    struct tm *time_str_tm;
                    time_str_tm = localtime(&time_now.tv_sec); /* local time, consistent with the local-time log filename */
                    fprintf(fd_temp, "%02i:%02i:%02i:%06li | %i | %lld\n"
                       , time_str_tm->tm_hour
                       , time_str_tm->tm_min
                       , time_str_tm->tm_sec
                       , time_now.tv_usec
                       , w_size
                       , curr_iters);
                    fflush(fd_temp);
                }
                //-----------------
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

done:
    /*write results to file*/
    MPI_Barrier(MPI_COMM_WORLD);
    write_results();

    //-----------------
    if(my_rank==master_rank && fd_temp!=NULL) fclose(fd_temp);
    //-----------------

    /*free allocated buffers*/
    free(durations);
    free(send_buf);
    free(recv_buf);
    data_free(send_dev);
    data_free(recv_dev);

    /*exit MPI library*/
    MPI_Finalize();
}
