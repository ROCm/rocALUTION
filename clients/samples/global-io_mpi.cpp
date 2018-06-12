#include <iostream>
#include <mpi.h>
#include <rocalution.hpp>

using namespace rocalution;

int main(int argc, char* argv[])
{
    if (argc < 2)
    { 
        std::cerr << argv[0] << " <parallelmanager> <matrix>" << std::endl;
        exit(1);
    }

    // Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;

    int rank;
    int num_procs;

    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &num_procs);

    if (num_procs < 2) {
        std::cerr << "Expecting more than 1 MPI process\n";
        MPI_Finalize();
        return -1;
    }

    // Disable OpenMP thread affinity
    set_omp_affinity_rocalution(false);

    // Initialize platform with rank and # of accelerator devices in the node
    init_rocalution(rank, 2);

    // Disable OpenMP
    set_omp_threads_rocalution(1);

    // Print platform
    info_rocalution();

    // Parallel Manager
    ParallelManager pm;

    // Initialize Parallel Manager
    pm.SetMPICommunicator(&comm);

    // Read Parallel Manager from file
    pm.ReadFileASCII(std::string(argv[1]));

    // Create global structures
    GlobalMatrix<double> mat(pm);
    GlobalVector<double> rhs(pm);
    GlobalVector<double> x(pm);
    GlobalVector<double> e(pm);

    // Move structures to accelerator, if available
    mat.MoveToAccelerator();
    rhs.MoveToAccelerator();
    x.MoveToAccelerator();
    e.MoveToAccelerator();

    // Read from file
    mat.ReadFileMTX(std::string(argv[2]));
    rhs.Allocate("rhs", mat.get_nrow());
    x.Allocate("x", mat.get_ncol());
    e.Allocate("sol", mat.get_ncol());

    e.Ones();
    mat.Apply(e, &rhs);
    x.Zeros();

    CG<GlobalMatrix<double>, GlobalVector<double>, double> ls;
    BlockJacobi<GlobalMatrix<double>, GlobalVector<double>, double> bj;
    FSAI<LocalMatrix<double>, LocalVector<double>, double> p;

    bj.Init(p);

    ls.SetPreconditioner(bj);
    ls.SetOperator(mat);
    ls.Build();
    ls.Verbose(1);

    mat.info();

    double time = rocalution_time();

    ls.Solve(rhs, &x);

    time = rocalution_time() - time;
    if (rank == 0)
    {
      std::cout << "Solving: " << time/1e6 << " sec" << std::endl;
    }

    e.ScaleAdd(-1.0, x);
    double nrm2 = e.Norm();
    if (rank == 0)
    {
        std::cout << "||e - x||_2 = " << nrm2 << std::endl;
    }

    ls.Clear();

    stop_rocalution();

    MPI_Finalize();

    return 0;
}
