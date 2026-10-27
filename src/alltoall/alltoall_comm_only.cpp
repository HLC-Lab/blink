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
#include <vector>
#include "common.h"



void all2all_memcpy(const void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm){

    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    
    int datatype_size;
    MPI_Type_size(sendtype, &datatype_size);

    const char* sbuf = static_cast<const char*>(sendbuf);
    char* rbuf = static_cast<char*>(recvbuf);

    // Copy local data directly (self-send)
    data_copy(rbuf + (size_t)rank * datatype_size * recvcount,
                sbuf + (size_t)rank * datatype_size * sendcount,
                (size_t)sendcount * datatype_size);

}

void custom_alltoall(const void* sendbuf, int sendcount, MPI_Datatype sendtype,
                     void* recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    int datatype_size;
    MPI_Type_size(sendtype, &datatype_size);

    const char* sbuf = static_cast<const char*>(sendbuf);
    char* rbuf = static_cast<char*>(recvbuf);

    std::vector<MPI_Request> requests;
    for (int i = 0; i < size; ++i) {
        if (i == rank) continue;

        MPI_Request req_recv;
        MPI_Request req_send;

        MPI_Isend(sbuf + i * datatype_size * sendcount, sendcount, sendtype, i, 0, comm, &req_send);
        MPI_Irecv(rbuf + i * datatype_size * recvcount, recvcount, recvtype, i, 0, comm, &req_recv);
        
        requests.push_back(req_send);
        requests.push_back(req_recv);
    }

    MPI_Waitall(static_cast<int>(requests.size()), requests.data(), MPI_STATUSES_IGNORE);
}

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

    send_buf_size=(size_t)msg_size*w_size;
    recv_buf_size=(size_t)msg_size*w_size;
    
    send_buf=(unsigned char*)malloc_align(send_buf_size);
    recv_buf=(unsigned char*)malloc_align(recv_buf_size);
    durations=(double *)malloc_align(sizeof(double)*max_samples);
    
    if(send_buf==NULL){
        fprintf(stderr,"Failed to allocate send_buf on rank %d\n",my_rank);
        MPI_Abort(MPI_COMM_WORLD, -1);
    } else if (recv_buf==NULL){
        fprintf(stderr,"Failed to allocate recv_buf on rank %d\n",my_rank);
        MPI_Abort(MPI_COMM_WORLD, -1);
    } else if (durations==NULL){
        fprintf(stderr,"Failed to allocate durations buffer on rank %d\n",my_rank);
        MPI_Abort(MPI_COMM_WORLD, -1);
    }
    
    /*fill send buffer with dummies*/
    for(size_t bi=0;bi<send_buf_size;bi++){
        send_buf[bi] = debug_mode ? (unsigned char)my_rank : 'a';
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
    double measure_total_time;
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
                /* comm_only metric: accumulate only the pure inter-rank transfer
                 * time across the granularity batch.  The local self-copy
                 * (all2all_memcpy) is intentionally outside the timed region, so
                 * this isolates communication.  This is a deliberately different
                 * measurement than the _b/_nb variants, which time the whole
                 * batched window as a single sample. */
                measure_total_time=0.0;
                for(i=0;i<measure_granularity;i++){
                    all2all_memcpy(send_dev, msg_size, MPI_BYTE, recv_dev, msg_size, MPI_BYTE, MPI_COMM_WORLD);
                    measure_start_time=MPI_Wtime();
                    custom_alltoall(send_dev, msg_size, MPI_BYTE, recv_dev, msg_size, MPI_BYTE, MPI_COMM_WORLD);
                    measure_total_time+=MPI_Wtime()-measure_start_time;
                }
                if (k >= warm_up_iters) record_duration(measure_total_time);
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
        data_fetch(recv_buf, recv_dev, recv_buf_size);
        int _r, _b, _ok = 1;
        for (_r = 0; _r < w_size && _ok; _r++)
            for (_b = 0; _b < msg_size && _ok; _b++)
                if (recv_buf[(size_t)_r * msg_size + _b] != (unsigned char)_r) _ok = 0;
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
    free(send_buf);
    free(recv_buf);
    data_free(send_dev);
    data_free(recv_dev);
    
    /*exit MPI library*/
    MPI_Finalize();
}

