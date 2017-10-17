#include <iostream>
#include <paralution.hpp>
#include <mpi.h>

using namespace paralution;

int main(int argc, char* argv[]) {

  if (argc < 4) { 
    std::cerr << argv[0] << " <parallelmanager> <matrix> <rhs> [Num threads]" << std::endl;
    exit(1);
  }

  MPI_Init(&argc, &argv);
  MPI_Comm comm = MPI_COMM_WORLD;

  int num_processes;
  int rank;

  MPI_Comm_size(comm, &num_processes);
  MPI_Comm_rank(comm, &rank);

  if (num_processes == 1) {
    std::cerr << "Expecting more than 1 MPI process\n";
    MPI_Finalize();
    exit(1);
  }

  set_omp_affinity(false);

  init_paralution(rank, 2);

  set_omp_threads_paralution(1);

  info_paralution();

  // Parallel Manager
  ParallelManager *pm = new ParallelManager;

  // Initialize Parallel Manager
  pm->SetMPICommunicator(&comm);

  // Read Parallel Manager from file
  pm->ReadFileASCII(std::string(argv[1]));

  // Create global structures
  GlobalMatrix<double> mat(*pm);
  GlobalVector<double> rhs(*pm);
  GlobalVector<double> x(*pm);

  mat.ReadFileMTX(std::string(argv[2]));
  rhs.ReadFileASCII(std::string(argv[3]));
  x.Allocate("x", mat.get_ncol());

  x.Zeros();

  CG<GlobalMatrix<double>, GlobalVector<double>, double> ls;
  BlockJacobi<GlobalMatrix<double>, GlobalVector<double>, double> bj;
  FSAI<LocalMatrix<double>, LocalVector<double>, double> p;

  bj.Init(p);

  ls.SetPreconditioner(bj);
  ls.SetOperator(mat);
  ls.Build();

  mat.MoveToAccelerator();
  rhs.MoveToAccelerator();
  x.MoveToAccelerator();
  ls.MoveToAccelerator();

  mat.info();

  double tick = paralution_time();

  ls.Solve(rhs, &x);

  double tack = paralution_time();
  std::cout << "Solver execution:" << (tack-tick)/1000000 << " sec" << std::endl;

  ls.Clear();

  stop_paralution();

  MPI_Finalize();

  return 0;

}
