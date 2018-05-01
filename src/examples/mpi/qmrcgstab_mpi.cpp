// *************************************************************************
//
// This example runs with 2 MPI processes (only).
//
// It constructs a 2D unit square with 2 sub-domains.
// The program distributes each domain (P1 and P2) by:
//
//        P1   |    P2
//  0  1  2  3 |  4  5  6  7
//  8  9 10 11 | 12 13 14 15
// 16 17 18 19 | 20 21 22 23
// 24 25 26 27 | 28 29 30 31
// 32 33 34 35 | 36 37 38 39
// 40 41 42 43 | 44 45 46 47
// 48 49 50 51 | 52 53 54 55
// 56 57 58 59 | 60 61 62 63
//
// Then, it constructs a 2D laplace operator and solves Lu=1,
// with initial guess 0. 
//
// The operator L has the following form:
//
//      -1
//   -1  4 -1
//      -1
//
// *************************************************************************

#include <iostream>
#include <cstdlib>
#include <rocalution.hpp>
#include <mpi.h>

// Adjust ndim for size
#define ndim 1000
#define n ndim*ndim

using namespace rocalution;

int main(int argc, char* argv[]) {

  MPI_Init(&argc, &argv);
  MPI_Comm comm = MPI_COMM_WORLD;

  int num_processes;
  int rank;

  MPI_Comm_size(comm, &num_processes);
  MPI_Comm_rank(comm, &rank);

  if (num_processes != 2) {
    std::cerr << "Expecting two MPI ranks\n";
    MPI_Finalize();
    exit(1);
  }

  set_omp_affinity(false);

  init_rocalution(rank, 2);

  set_omp_threads_rocalution(1);

  info_rocalution();

  // Parallel Manager
  ParallelManager *pm = new ParallelManager;

  // Initialize Parallel Manager
  pm->SetMPICommunicator(&comm);

  // Determine global and local matrix and vector sizes
  int nboundary = sqrt(n);
  pm->SetGlobalSize(n);
  pm->SetLocalSize(n/num_processes);

  // Fill the boundary indices of the 2D domain
  int *BoundaryIndex = new int[nboundary];

  for (int i=0; i<nboundary; ++i) {
    if(rank==0) BoundaryIndex[i] = (i+1)*nboundary/num_processes-1;
    if(rank==1) BoundaryIndex[i] = (i+1)*nboundary/num_processes-nboundary/num_processes;
  }

  // Pass the boundary indices to the parallel manager
  pm->SetBoundaryIndex(nboundary, BoundaryIndex);

  // Specify senders and receivers for the communication pattern
  int *recv = new int[num_processes-1];
  int *sender = new int[num_processes-1];
  if (rank == 0) {
    recv[0] = 1;
    sender[0] = 1;
  }
  if (rank == 1) {
    recv[0] = 0;
    sender[0] = 0;
  }

  // Determine the offsets for each rank
  int *recv_offset = new int[num_processes];
  int *sender_offset = new int[num_processes];

  recv_offset[0] = 0;
  recv_offset[1] = nboundary;
  sender_offset[0] = 0;
  sender_offset[1] = nboundary;

  // Pass the boundary and ghost communication pattern to the parallel manager
  pm->SetReceivers(num_processes-1, recv, recv_offset);
  pm->SetSenders(num_processes-1, sender, sender_offset);

  GlobalMatrix<double> mat(*pm);
  GlobalVector<double> rhs(*pm);
  GlobalVector<double> x(*pm);

  // Compute the local and ghost non-zero entries
  int local_nnz = n/num_processes*5-nboundary*2-nboundary/num_processes*2;
  int ghost_nnz = nboundary;

  rhs.Allocate("rhs", n);
  x.Allocate("x", n);

  rhs.Ones();
  x.Zeros();

  int *local_row_offset = NULL;
  int *local_col = NULL;
  double *local_val = NULL;

  allocate_host(n/num_processes+1, &local_row_offset);
  allocate_host(local_nnz, &local_col);
  allocate_host(local_nnz, &local_val);

  int *ghost_row_offset = NULL;
  int *ghost_col = NULL;
  double *ghost_val = NULL;

  allocate_host(n/num_processes+1, &ghost_row_offset);
  allocate_host(ghost_nnz, &ghost_col);
  allocate_host(ghost_nnz, &ghost_val);

  // Assembly

  // Mesh example for n = 64:
  //        P1   |    P2
  //  0  1  2  3 |  4  5  6  7
  //  8  9 10 11 | 12 13 14 15
  // 16 17 18 19 | 20 21 22 23
  // 24 25 26 27 | 28 29 30 31
  // 32 33 34 35 | 36 37 38 39
  // 40 41 42 43 | 44 45 46 47
  // 48 49 50 51 | 52 53 54 55
  // 56 57 58 59 | 60 61 62 63

  int dim = sqrt(n);
  int nnz = 0;
  int k = 0;

  // Fill local arrays
  if (rank == 0) {
    for (int i=0; i<dim; ++i) {
      for (int j=0; j<dim/num_processes; ++j) {

        int idx = i*dim/num_processes+j;
        local_row_offset[k] = nnz;
        ++k;

        // if no upper boundary element, connect with upper neighbor
        if (i != 0) {
          local_col[nnz] = idx - dim/num_processes;
          local_val[nnz] = -1.0;
          ++nnz;
        }

        // if no left boundary element, connect with left neighbor
        if (j != 0) {
          local_col[nnz] = idx - 1;
          local_val[nnz] = -1.0;
          ++nnz;
        }

        // element itself
        local_col[nnz] = idx;
        local_val[nnz] = 4.0;
        ++nnz;

        // if no right boundary element, connect with right neighbor
        if (j != dim/num_processes - 1) {
          local_col[nnz] = idx + 1;
          local_val[nnz] = -1.0;
          ++nnz;
        }

        // if no lower boundary element, connect with lower neighbor
        if (i != dim - 1) {
          local_col[nnz] = idx + dim/num_processes;
          local_val[nnz] = -1.0;
          ++nnz;
        }

      }
    }

    local_row_offset[k] = nnz;

    nnz = 0;
    k = 0;

    // Fill ghost arrays
    for (int i=0; i<dim; ++i) {
      for (int j=0; j<dim/num_processes; ++j) {

        int idx = i*dim/num_processes+j;
        ghost_row_offset[k] = nnz;
        ++k;

        // Boundary Values
        if (idx == BoundaryIndex[i]) {
          ghost_col[nnz] = i;
          ghost_val[nnz] = -1;
          ++nnz;
        }

      }
    }

    ghost_row_offset[n/num_processes] = nnz;

  }

  if (rank == 1) {
    for (int i=0; i<dim; ++i) {
      for (int j=dim/num_processes; j<dim; ++j) {

        int idx = i*dim/num_processes+j-dim/num_processes;
        local_row_offset[k] = nnz;
        ++k;

        // if no upper boundary element, connect with upper neighbor
        if (i != 0) {
          local_col[nnz] = idx - dim/num_processes;
          local_val[nnz] = -1.0;
          ++nnz;
        }

        // if no left boundary element, connect with left neighbor
        if (j != dim/num_processes) {
          local_col[nnz] = idx - 1;
          local_val[nnz] = -1.0;
          ++nnz;
        }

        // element itself
        local_col[nnz] = idx;
        local_val[nnz] = 4.0;
        ++nnz;

        // if no right boundary element, connect with right neighbor
        if (j != dim - 1) {
          local_col[nnz] = idx + 1;
          local_val[nnz] = -1.0;
          ++nnz;
        }

        // if no lower boundary element, connect with lower neighbor
        if (i != dim - 1) {
          local_col[nnz] = idx + dim/num_processes;
          local_val[nnz] = -1.0;
          ++nnz;
        }

      }
    }

    local_row_offset[n/num_processes] = nnz;

    nnz = 0;
    k = 0;

    // Fill ghost arrays
    for (int i=0; i<dim; ++i) {
      for (int j=dim/num_processes; j<dim; ++j) {

        int idx = i*dim/num_processes+j-dim/num_processes;
        ghost_row_offset[k] = nnz;
        ++k;

        // Boundary Values
        if (idx == BoundaryIndex[i]) {
          ghost_col[nnz] = i;
          ghost_val[nnz] = -1;
          ++nnz;
        }

      }
    }

    ghost_row_offset[n/num_processes] = nnz;

  }

  // Set Matrix
  mat.SetDataPtrCSR(&local_row_offset, &local_col, &local_val,
                    &ghost_row_offset, &ghost_col, &ghost_val,
                    "mat", local_nnz, ghost_nnz);

  x.Zeros();
  rhs.Ones();

  QMRCGStab<GlobalMatrix<double>, GlobalVector<double>, double> ls;
  BlockJacobi<GlobalMatrix<double>, GlobalVector<double>, double> bj;
  ILUT<LocalMatrix<double>, LocalVector<double>, double> p;

  bj.Init(p);

  ls.SetPreconditioner(bj);
  ls.SetOperator(mat);
  ls.Build();
  ls.Verbose(1);

  mat.MoveToAccelerator();
  rhs.MoveToAccelerator();
  x.MoveToAccelerator();
  ls.MoveToAccelerator();

  mat.info();

  double tick = rocalution_time();

  ls.Solve(rhs, &x);

  double tack = rocalution_time();
  std::cout << "Solving:" << (tack-tick)/1000000 << " sec" << std::endl;

  ls.Clear();

  delete [] recv;
  delete [] sender;
  delete [] BoundaryIndex;
  delete [] recv_offset;
  delete [] sender_offset;
  delete pm;

  stop_rocalution();

  MPI_Finalize();

  return 0;

}

