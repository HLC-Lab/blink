#include <mpi.h>
#include "common.h"

/* No Data buffers: -gpu is accepted and has no effect. */
int benchmark_has_data_buffers = 0;

int main(int argc, char** argv){
    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD, &w_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    gpu_tool_args(argc, argv);   /* ignores every argument except -gpu (which only warns) */
    MPI_Finalize();
}
