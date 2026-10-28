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
"  -dimx <int>                   first dim of the 2D Cartesian grid (default 0 = auto via MPI_Dims_create)\n"
"                                if non-zero, must divide w_size; w_size/dimx becomes the Y dim\n"
"  -periodic                     enable wrap-around (toroidal) boundaries on both dims\n";

int main(int argc, char** argv){

    /*init MPI world*/
    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD, &w_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    /*register signal handler*/
    install_shutdown_handler();

    int dimx=0; /*0 means auto-factor via MPI_Dims_create*/
    bool periodic=false;

    /*parse command line*/
    int i, k;
    argc = parse_common_args(argc, argv);
    for (i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-dimx") == 0) {
            dimx = atoi(arg_value(argc, argv, &i));
        } else if (strcmp(argv[i], "-periodic") == 0) {
            periodic = true;
        } else {
            if (my_rank == master_rank) {
                fprintf(stderr, "Unknown argument: %s\n", argv[i]);
                MPI_Abort(MPI_COMM_WORLD, -1);
            }
        }
    }

    /*build 2D Cartesian communicator*/
    int dims[2]={0,0};
    int periods[2]={0,0};

    if(dimx>0){
        if(w_size%dimx!=0){
            if(my_rank==master_rank){
                fprintf(stderr,"dimx (%d) does not divide w_size (%d)\n",dimx,w_size);
                MPI_Abort(MPI_COMM_WORLD, -1);
            }
        }
        dims[0]=dimx;
        dims[1]=w_size/dimx;
    }else{
        MPI_Dims_create(w_size,2,dims);
    }

    if(periodic){
        periods[0]=1;
        periods[1]=1;
    }

    MPI_Comm cart_comm;
    MPI_Cart_create(MPI_COMM_WORLD,2,dims,periods,0,&cart_comm);

    /*get 4-neighbor ranks (MPI_PROC_NULL for missing boundary neighbors)*/
    /*direction 0 = rows: north = row-1, south = row+1*/
    /*direction 1 = cols: west = col-1, east = col+1*/
    int north, south, west, east;
    MPI_Cart_shift(cart_comm,0,1,&north,&south);
    MPI_Cart_shift(cart_comm,1,1,&west,&east);

    int neighbors[4]={north,south,west,east};

    /*pin to core*/
    /*cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(1, &mask);
    sched_setaffinity(0, sizeof(mask), &mask);*/

    /*allocate buffers*/
    /*4 directions, each of size msg_size; granularity batches them*/
    size_t send_buf_size, recv_buf_size;
    unsigned char *send_buf;
    unsigned char *recv_buf;
    MPI_Request *send_requests;
    MPI_Request *recv_requests;

    send_buf_size=msg_size;
    recv_buf_size=(size_t)4*measure_granularity*msg_size;

    send_buf=(unsigned char*)malloc_align(send_buf_size);
    recv_buf=(unsigned char*)malloc_align(recv_buf_size);
    durations=(double *)malloc_align(sizeof(double)*max_samples);
    send_requests=(MPI_Request*)malloc_align(sizeof(MPI_Request)*4*measure_granularity);
    recv_requests=(MPI_Request*)malloc_align(sizeof(MPI_Request)*4*measure_granularity);


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
            printf("2D stencil with %d processes (%dx%d grid), periodic: %s, msg-size: %d, test iterations: endless.\n"
                    ,w_size,dims[0],dims[1],(periodic?"true":"false"),msg_size);
        }else{
            printf("2D stencil with %d processes (%dx%d grid), periodic: %s, msg-size: %d, test iterations: %d.\n"
                    ,w_size,dims[0],dims[1],(periodic?"true":"false"),msg_size,max_iters);
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
    int d;

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
                    /*exchange with all 4 neighbors; MPI_PROC_NULL ops complete immediately*/
                    for(d=0;d<4;d++){
                        MPI_Irecv(&recv_dev[(4*i+d)*msg_size],msg_size,MPI_BYTE,
                                  neighbors[d],antideadlock_tag,cart_comm,&recv_requests[4*i+d]);
                        MPI_Isend(send_dev,msg_size,MPI_BYTE,
                                  neighbors[d],antideadlock_tag,cart_comm,&send_requests[4*i+d]);
                    }
                    antideadlock_tag++;
                }
                MPI_Waitall(4*measure_granularity,send_requests,MPI_STATUSES_IGNORE);
                MPI_Waitall(4*measure_granularity,recv_requests,MPI_STATUSES_IGNORE);
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
        data_fetch(recv_buf, recv_dev, recv_buf_size);
        int _d, _b, _ok = 1;
        const int _dirs[4] = {north, south, west, east};
        for (_d = 0; _d < 4 && _ok; _d++) {
            if (_dirs[_d] < 0) continue; /* MPI_PROC_NULL — no neighbour, no data */
            for (_b = 0; _b < msg_size && _ok; _b++)
                if (recv_buf[(size_t)_d * msg_size + _b] != (unsigned char)_dirs[_d]) _ok = 0;
        }
        printf("DEBUG rank=%d nprocs=%d north=%d south=%d west=%d east=%d check=%s\n",
               my_rank, w_size, north, south, west, east, _ok ? "OK" : "FAIL");
        fflush(stdout);
        MPI_Barrier(MPI_COMM_WORLD);
    }
done:
    /*write results to file*/
    MPI_Barrier(MPI_COMM_WORLD);
    write_results();

    /*free allocated buffers*/
    MPI_Comm_free(&cart_comm);
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
