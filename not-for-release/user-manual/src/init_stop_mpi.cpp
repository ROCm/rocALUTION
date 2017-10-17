#include <paralution.hpp>
#include <mpi.h>

using namespace paralution;

int main(int argc, char* argv[]) {

  MPI_Init(&argc, &argv);
  MPI_Comm comm = MPI_COMM_WORLD;

  int num_processes;
  int rank;
  MPI_Comm_size(comm, &num_processes);
  MPI_Comm_rank(comm, &rank);

  init_paralution(rank, 2);

  info_paralution();

  // ...

  stop_paralution();

}
