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


static inline int copy_buffer_different_dt (const void *input_buffer, size_t scount,
                                            const MPI_Datatype sdtype, void *output_buffer,
                                            size_t rcount, const MPI_Datatype rdtype) {
  if (input_buffer == NULL || output_buffer == NULL || scount <= 0 || rcount <= 0) {
    return MPI_ERR_UNKNOWN;
  }

  int sdtype_size;
  MPI_Type_size(sdtype, &sdtype_size);
  int rdtype_size;
  MPI_Type_size(rdtype, &rdtype_size);

  size_t s_size = (size_t) sdtype_size * scount;
  size_t r_size = (size_t) rdtype_size * rcount;

  if (r_size < s_size) {
    data_copy(output_buffer, input_buffer, r_size); // Copy as much as possible
    return MPI_ERR_TRUNCATE;      // Indicate truncation
  }

  data_copy(output_buffer, input_buffer, s_size);        // Perform the memory copy

  return MPI_SUCCESS;
}


void allgather_memcpy(const void *sbuf, size_t scount, MPI_Datatype sdtype, void* rbuf, size_t rcount, MPI_Datatype rdtype, MPI_Comm comm){

  int rank;
  MPI_Aint rlb, rext;
  char *tmpsend = NULL, *tmprecv = NULL;

  MPI_Comm_rank(comm, &rank);

  MPI_Type_get_extent(rdtype, &rlb, &rext);

  tmprecv = (char*) rbuf + (size_t)rank * rcount * rext;
  if (MPI_IN_PLACE != sbuf) {
    tmpsend = (char*) sbuf;
    copy_buffer_different_dt(tmpsend, scount, sdtype, tmprecv, rcount, rdtype);
  }
}


void allgather_ring(const void *sbuf, size_t scount, MPI_Datatype sdtype,
                   void* rbuf, size_t rcount, MPI_Datatype rdtype, MPI_Comm comm) {

  int rank, size, sendto, recvfrom, i, recvdatafrom, senddatafrom;
  MPI_Aint rlb, rext;
  char *tmpsend = NULL, *tmprecv = NULL;

  MPI_Comm_size(comm, &size);
  MPI_Comm_rank(comm, &rank);

  MPI_Type_get_extent(rdtype, &rlb, &rext);

  sendto = (rank + 1) % size;
  recvfrom  = (rank - 1 + size) % size;

  for (i = 0; i < size - 1; i++) {

    recvdatafrom = (rank - i - 1 + size) % size;
    senddatafrom = (rank - i + size) % size;

    tmprecv = (char*)rbuf + recvdatafrom * rcount * rext;
    tmpsend = (char*)rbuf + senddatafrom * rcount * rext;

    MPI_Sendrecv(tmpsend, rcount, rdtype, sendto, 0,
                       tmprecv, rcount, rdtype, recvfrom, 0,
                       comm, MPI_STATUS_IGNORE);

  }
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
    size_t msg_size_ints;
    size_t send_buf_size, recv_buf_size;
    int *send_buf;
    int *recv_buf;
    
    if(msg_size%sizeof(int)!=0){
        if(my_rank==master_rank)
            fprintf(stderr, "Msg-size (%d) must be divisible by size of int (%zu)\n",msg_size,sizeof(int));
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    /* -msgsize is the per-rank contribution (matches allgather_b/nb); the full
     * gathered result is w_size * msg_size bytes. */
    send_buf_size=msg_size;
    msg_size_ints=(size_t)msg_size/sizeof(int);
    recv_buf_size=(size_t)w_size*msg_size;
    
    send_buf=(int*)malloc_align(send_buf_size);
    recv_buf=(int*)malloc_align(recv_buf_size);
    durations=(double *)malloc_align(sizeof(double)*max_samples);
    
    
    /*fill send buffer with dummies*/
    for(i=0;i<msg_size_ints;i++){
        send_buf[i] = debug_mode ? my_rank : 1;
    }

    /*GPU mode: MPI works on device mirrors; in host mode the mirrors are the host buffers*/
    int *send_dev = (int*)data_mirror(send_buf, send_buf_size);
    int *recv_dev = (int*)data_mirror(recv_buf, recv_buf_size);

    
    /*print basic info to stdout*/
    if(my_rank==master_rank){
        if(endless){
            printf("Allgather with %d processes, msg-size: %d, test iterations: endless.\n"
                    ,w_size,msg_size);
        }else{
            printf("Allgather with %d processes, msg-size: %d, test iterations: %d.\n"
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
                 * (allgather_memcpy) is intentionally outside the timed region, so
                 * this isolates communication.  This is a deliberately different
                 * measurement than the _b/_nb variants, which time the whole
                 * batched window as a single sample. */
                measure_total_time=0.0;
                for(i=0;i<measure_granularity;i++){
                    allgather_memcpy(send_dev, msg_size_ints, MPI_INT, recv_dev, msg_size_ints, MPI_INT, MPI_COMM_WORLD);
                    measure_start_time=MPI_Wtime();
                    allgather_ring(send_dev, msg_size_ints, MPI_INT, recv_dev, msg_size_ints, MPI_INT, MPI_COMM_WORLD);
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
        size_t _r, _b;
        int _ok = 1;
        for (_r = 0; _r < (size_t)w_size && _ok; _r++)
            for (_b = 0; _b < msg_size_ints && _ok; _b++)
                if (recv_buf[_r * msg_size_ints + _b] != (int)_r) _ok = 0;
        printf("DEBUG rank=%d nprocs=%d bytes=%zu check=%s\n",
               my_rank, w_size, (size_t)send_buf_size, _ok ? "OK" : "FAIL");
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

