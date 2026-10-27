#include <mpi.h>
#include "common.h"

/* No Data buffers: -gpu is accepted and has no effect. */
int benchmark_has_data_buffers = 0;

int main(int argc, char** argv){
    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD, &w_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    parse_common_args(argc, argv);
    MPI_Finalize();
}
