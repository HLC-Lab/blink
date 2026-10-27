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
"  -mode <offpair|rpair|rot|perm>   pairing strategy (default: offpair)\n"
"      offpair   symmetric offset pairs (uses -offset)\n"
"      rpair     random reciprocal pairing (uses -seed)\n"
"      rot       rotation: rank r -> (r+offset) mod w_size; non-reciprocal\n"
"      perm      random permutation; non-reciprocal\n"
"  -offset <int>                 offset for offpair/rot modes (default w_size/2)\n";

int main(int argc, char** argv){

    /*init MPI world*/
    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD, &w_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    
    /*register signal handler*/
    install_shutdown_handler();

    const char *comm_mode="offpair";
    int target_offset=w_size/2;

    /*parse command line*/
    int i, k;
    argc = parse_common_args(argc, argv);
    for (i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-offset") == 0) {
            target_offset = atoi(arg_value(argc, argv, &i));
        } else if (strcmp(argv[i], "-mode") == 0) {
            comm_mode = arg_value(argc, argv, &i);
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
    MPI_Request *send_requests;

    send_buf_size=msg_size;
    recv_buf_size=(size_t)measure_granularity*msg_size;
    
    send_buf=(unsigned char*)malloc_align(send_buf_size);
    recv_buf=(unsigned char*)malloc_align(recv_buf_size);
    targets=(int*)malloc_align(sizeof(int)*w_size);
    durations=(double *)malloc_align(sizeof(double)*max_samples);
    recv_requests=(MPI_Request*)malloc_align(sizeof(MPI_Request)*measure_granularity);
    send_requests=(MPI_Request*)malloc_align(sizeof(MPI_Request)*measure_granularity);
    
    
    /*fill send buffer: rank-as-payload under -debug so partners can verify*/
    for(i=0;i<send_buf_size;i++){
        send_buf[i] = debug_mode ? (unsigned char)my_rank : 'a';
    }

    /*GPU mode: MPI works on device mirrors; in host mode the mirrors are the host buffers*/
    unsigned char *send_dev = (unsigned char*)data_mirror(send_buf, send_buf_size);
    unsigned char *recv_dev = (unsigned char*)data_mirror(recv_buf, recv_buf_size);
    
    /*setup target mode*/
    if(strcmp(comm_mode,"perm")==0){
        for(i=0;i<w_size;i++){
            targets[i]=i;
        }
        permute(targets,w_size);
    }else if(strcmp(comm_mode,"rot")==0){
        for(i=0;i<w_size;i++){
            targets[i]=mod(i+target_offset,w_size);
        }
    }else if(strcmp(comm_mode,"offpair")==0){
        offset_pairs(targets,w_size,target_offset);
    }else if(strcmp(comm_mode,"rpair")==0){
        random_pairs(targets,w_size);
    }else{
        if(my_rank==master_rank){
            fprintf(stderr,"Unknown communication mode: %s\n",comm_mode);
            MPI_Abort(MPI_COMM_WORLD, -1);
        }
    }
    
    /*print basic info to stdout*/
    if(my_rank==master_rank){
        if(endless){
            printf("Pairwise with %d processes, mode: %s, msg-size: %d, test iterations: endless.\n"
                    ,w_size,comm_mode,msg_size);
        }else{
            printf("Pairwise with %d processes, mode: %s, msg-size: %d, test iterations: %d.\n"
                    ,w_size,comm_mode,msg_size,max_iters);
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
                    MPI_Irecv(&recv_dev[i*msg_size],msg_size,MPI_BYTE,MPI_ANY_SOURCE
                        ,MPI_ANY_TAG,MPI_COMM_WORLD,&recv_requests[i]);
                    MPI_Isend(send_dev,msg_size,MPI_BYTE,targets[my_rank]
                        ,my_rank,MPI_COMM_WORLD,&send_requests[i]);
                }
                MPI_Waitall(measure_granularity,send_requests,MPI_STATUSES_IGNORE);
                MPI_Waitall(measure_granularity,recv_requests,MPI_STATUSES_IGNORE);
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

    /* debug: deterministic post-loop exchange to verify partner identity and
     * payload integrity.  Send to targets[my_rank] and receive from this rank's
     * inverse source (the rank whose target is me).  These differ in the
     * non-reciprocal modes (perm/rot), so a non-blocking exchange is used to
     * avoid deadlock.  Barrier first to drain any measurement-loop ANY_TAG
     * receives so they don't consume our debug-tagged sends.                */
    if (debug_mode) {
        MPI_Barrier(MPI_COMM_WORLD);
        int dst = targets[my_rank];
        int src = -1, _t;
        for (_t = 0; _t < w_size; _t++) if (targets[_t] == my_rank) { src = _t; break; }
        int _ok = 1, _b;
        if (dst != my_rank && src >= 0) {
            MPI_Request reqs[2];
            MPI_Irecv(recv_dev, msg_size, MPI_BYTE, src, 0xD, MPI_COMM_WORLD, &reqs[0]);
            MPI_Isend(send_dev, msg_size, MPI_BYTE, dst, 0xD, MPI_COMM_WORLD, &reqs[1]);
            MPI_Waitall(2, reqs, MPI_STATUSES_IGNORE);
            data_fetch(recv_buf, recv_dev, recv_buf_size);
            for (_b = 0; _b < msg_size && _ok; _b++)
                if (recv_buf[_b] != (unsigned char)src) _ok = 0;
        }
        printf("DEBUG rank=%d nprocs=%d partner=%d check=%s\n",
               my_rank, w_size, dst, _ok ? "OK" : "FAIL");
        fflush(stdout);
        MPI_Barrier(MPI_COMM_WORLD);
    }
done:
    /*write results to file*/
    MPI_Barrier(MPI_COMM_WORLD);
    write_results();
    
    /*free allocated buffers*/
    free(targets);
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

