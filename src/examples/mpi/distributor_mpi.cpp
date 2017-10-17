#include <iostream>
#include <fstream>
#include <algorithm>
#include <vector>
#include <map>
#include <mpi.h>
#include <paralution.hpp>

#define ValueType double

using namespace paralution;

int main(int argc, char* argv[]) {

  MPI_Init(&argc, &argv);
  MPI_Comm comm = MPI_COMM_WORLD;

  int rank;
  int num_procs;

  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &num_procs);

  if (num_procs < 2) {
    std::cerr << "Expecting at least 2 MPI processes" << std::endl;
    return -1;
  }

  if (argc < 4) { 
    std::cerr << argv[0] << " <global_matrix> <manager> [output] <distributed_matrix> [output]" << std::endl;
    return -1;
  }

  // Initialize platform with rank and # of accelerator devices in the node
  set_omp_affinity(false);

  init_paralution(rank, 2);

  // Disable OpenMP
  set_omp_threads_paralution(1);

  // Print platform
  info_paralution();

  // Load undistributed matrix
  LocalMatrix<ValueType> undis_mat;
  undis_mat.ReadFileMTX(argv[1]);

  size_t global_nrow = undis_mat.get_nrow();

  int *global_row_offset = NULL;
  int *global_col = NULL;
  ValueType *global_val = NULL;

  undis_mat.LeaveDataPtrCSR(&global_row_offset, &global_col, &global_val);

  // Compute local matrix sizes
  int *local_size = new int[num_procs];

  for (int i=0; i<num_procs; ++i) {
    local_size[i] = global_nrow / num_procs;
  }

  if (global_nrow % num_procs != 0) {
    for (size_t i=0; i<global_nrow % num_procs; ++i) {
      ++local_size[i];
    }
  }

  // Compute index offsets
  int *index_offset = new int[num_procs+1];
  index_offset[0] = 0;
  for (int i=0; i<num_procs; ++i) {
    index_offset[i+1] = index_offset[i] + local_size[i];
  }

  // Read sub matrix - row_offset
  int local_nrow = local_size[rank];
  int *local_row_offset = new int[local_nrow+1];

  for (int i=index_offset[rank], k=0; k<local_nrow+1; ++i, ++k) {
    local_row_offset[k] = global_row_offset[i];
  }

  // Read sub matrix - col and val
  int local_nnz = local_row_offset[local_nrow] - local_row_offset[0];
  int *local_col = new int[local_nnz];
  ValueType *local_val = new ValueType[local_nnz];

  for (int i=local_row_offset[0], k=0; k<local_nnz; ++i, ++k) {
    local_col[k] = global_col[i];
    local_val[k] = global_val[i];
  }

  // Shift row_offset entries
  int shift = local_row_offset[0];
  for (int i=0; i<local_nrow+1; ++i) {
    local_row_offset[i] -= shift;
  }

  int interior_nnz = 0;
  int ghost_nnz = 0;
  int boundary_nnz = 0;
  int neighbors = 0;

  std::vector<std::vector<int> > boundary(num_procs, std::vector<int>());
  std::vector<bool> neighbor(num_procs, false);
  std::vector<std::map<int, bool> > checked(num_procs, std::map<int, bool>());

  for (int i=0; i<local_nrow; ++i) {
    for (int j=local_row_offset[i]; j<local_row_offset[i+1]; ++j) {

      // Interior point
      if (local_col[j] >= index_offset[rank] && local_col[j] < index_offset[rank+1]) {
        ++interior_nnz;
      } else {
        // Boundary point above current process
        if (local_col[j] < index_offset[rank]) {
          // Loop over ranks above current process
          for (int r=rank-1; r>=0; --r) {
            // Check if boundary belongs to rank r
            if (local_col[j] >= index_offset[r] && local_col[j] < index_offset[r+1]) {
              // Add boundary point to rank r if it has not been added yet
              if (!checked[r][i+index_offset[rank]]) {
                boundary[r].push_back(i+index_offset[rank]);
                neighbor[r] = true;
                ++boundary_nnz;
                checked[r][i+index_offset[rank]] = true;
              }
              ++ghost_nnz;
              // Rank for current boundary point local_col[j] has been found -- continue with next boundary point
              break;
            }
          }
        }

        // boundary point below current process
        if (local_col[j] >= index_offset[rank+1]) {
          // Loop over ranks above current process
          for (int r=rank+1; r<num_procs; ++r) {
            // Check if boundary belongs to rank r
            if (local_col[j] >= index_offset[r] && local_col[j] < index_offset[r+1]) {
              // Add boundary point to rank r if it has not been added yet
              if (!checked[r][i+index_offset[rank]]) {
                boundary[r].push_back(i+index_offset[rank]);
                neighbor[r] = true;
                ++boundary_nnz;
                checked[r][i+index_offset[rank]] = true;
              }
              ++ghost_nnz;
              // Rank for current boundary point local_col[j] has been found -- continue with next boundary point
              break;
            }
          }
        }
      }

    }
  }

  for (int i=0; i<num_procs; ++i) {
    if (neighbor[i] == true) {
      ++neighbors;
    }
  }

  std::vector<MPI_Request> mpi_req(neighbors*2);
  int n = 0;
  // Array to hold boundary size for each interface
  int *boundary_size = new int[neighbors];

  // MPI receive boundary sizes
  for (int i=0; i<num_procs; ++i) {
    // If neighbor receive from rank i is expected...
    if (neighbor[i] == true) {
      // Receive size of boundary from rank i to current rank
      MPI_Irecv(&(boundary_size[n]), 1, MPI_INT, i, 0, comm, &mpi_req[n]);
      ++n;
    }
  }

  // MPI send boundary sizes
  for (int i=0; i<num_procs; ++i) {
    // Send required if boundary for rank i available
    if (boundary[i].size() > 0) {
      int size = boundary[i].size();
      // Send size of boundary from current rank to rank i
      MPI_Isend(&size, 1, MPI_INT, i, 0, comm, &mpi_req[n]);
      ++n;
    }
  }
  // Wait to finish communication
  MPI_Waitall(n-1, &(mpi_req[0]), MPI_STATUSES_IGNORE);

  n = 0;
  // Array to hold boundary offset for each interface
  int k = 0;
  int *recv_offset = new int[neighbors+1];
  int *send_offset = new int[neighbors+1];
  recv_offset[0] = 0;
  send_offset[0] = 0;
  for (int i=0; i<neighbors; ++i) {
    recv_offset[i+1] = recv_offset[i] + boundary_size[i];
  }

  for (int i=0; i<num_procs; ++i) {
    if (neighbor[i] == true) {
      send_offset[k+1] = send_offset[k] + boundary[i].size();
      ++k;
    }
  }

  // Array to hold boundary for each interface
  int **local_boundary = new int*[neighbors];
  for (int i=0; i<neighbors; ++i) {
    local_boundary[i] = new int[boundary_size[i]];
  }

  // MPI receive boundary
  for (int i=0; i<num_procs; ++i) {
    // If neighbor receive from rank i is expected...
    if (neighbor[i] == true) {
      // Receive boundary from rank i to current rank
      MPI_Irecv(local_boundary[n], boundary_size[n], MPI_INT, i, 0, comm, &mpi_req[n]);
      ++n;
    }
  }

  // MPI send boundary
  for (int i=0; i<num_procs; ++i) {
    // Send required if boundary for rank i is available
    if (boundary[i].size() > 0) {
      // Send boundary from current rank to rank i
      MPI_Isend(&(boundary[i][0]), boundary[i].size(), MPI_INT, i, 0, comm, &mpi_req[n]);
      ++n;
    }
  }

  // Wait to finish communication
  MPI_Waitall(n-1, &(mpi_req[0]), MPI_STATUSES_IGNORE);

  // Total boundary size
  int nnz_boundary = 0;
  for (int i=0; i<neighbors; ++i) {
    nnz_boundary += boundary_size[i];
  }

  // Create local boundary index array
  k = 0;
  int *bnd = new int[boundary_nnz];

  for (int i=0; i<num_procs; ++i) {
    for (unsigned int j=0; j<boundary[i].size(); ++j) {
      bnd[k] = boundary[i][j]-index_offset[rank];
      ++k;
    }
  }

  // Create boundary index array
  int *boundary_index = new int[nnz_boundary];

  k = 0;
  for (int i=0; i<neighbors; ++i) {
    for (int j=0; j<boundary_size[i]; ++j) {
      boundary_index[k] = local_boundary[i][j];
      ++k;
    }
  }

  // Create map with boundary index relations
  std::map<int, int> boundary_map;

  for (int i=0; i<nnz_boundary; ++i) {
    boundary_map[boundary_index[i]] = i;
  }

  // Build up ghost and interior matrix
  int *ghost_row = new int[ghost_nnz];
  int *ghost_col = new int[ghost_nnz];
  ValueType *ghost_val = new ValueType[ghost_nnz];

  int *row_offset = new int[local_nrow+1];
  int *col = new int[interior_nnz];
  ValueType *val = new ValueType[interior_nnz];

  row_offset[0] = 0;
  k = 0;
  int l = 0;
  for (int i=0; i<local_nrow; ++i) {
    for (int j=local_row_offset[i]; j<local_row_offset[i+1]; ++j) {

      // Boundary point -- create ghost part
      if (local_col[j] < index_offset[rank] || local_col[j] >= index_offset[rank+1]) {

        ghost_row[k] = i;
        ghost_col[k] = boundary_map[local_col[j]];
        ghost_val[k] = local_val[j];
        ++k;

      } else {
        // Interior point -- create interior part
        int c = local_col[j] - index_offset[rank];

        col[l] = c;
        val[l] = local_val[j];
        ++l;
      }

    }

    row_offset[i+1] = l;

  }

  int *recv = new int[neighbors];
  int *sender = new int[neighbors];

  int nbc = 0;
  for (int i=0; i<num_procs; ++i) {
    if (neighbor[i] == true) {
      recv[nbc] = i;
      sender[nbc] = i;
      ++nbc;
    }
  }

  ParallelManager manager;

  manager.SetMPICommunicator(&comm);
  manager.SetGlobalSize(global_nrow);
  manager.SetLocalSize(local_size[rank]);
  manager.SetBoundaryIndex(boundary_nnz, bnd);
  manager.SetReceivers(neighbors, recv, recv_offset);
  manager.SetSenders(neighbors, sender, send_offset);

  GlobalMatrix<ValueType> mat(manager);
  GlobalVector<ValueType> b(manager);
  GlobalVector<ValueType> x(manager);
  GlobalVector<ValueType> e(manager);

  mat.SetLocalDataPtrCSR(&row_offset, &col, &val, "mat", interior_nnz);
  mat.SetGhostDataPtrCOO(&ghost_row, &ghost_col, &ghost_val, "ghost", ghost_nnz);

  mat.WriteFileCSR(argv[3]);
  manager.WriteFileASCII(argv[2]);

  b.Allocate("rhs", mat.get_ncol());
  x.Allocate("x", mat.get_nrow());
  e.Allocate("e", mat.get_nrow());

  e.Ones();
  mat.Apply(e, &b);
  x.Zeros();

  CG<GlobalMatrix<ValueType>, GlobalVector<ValueType>, ValueType> ls;
  Jacobi<GlobalMatrix<ValueType>, GlobalVector<ValueType>, ValueType> p;

  ls.SetOperator(mat);
  ls.SetPreconditioner(p);

  ls.Build();
//  ls.Verbose(2);

  mat.MoveToAccelerator();
  b.MoveToAccelerator();
  x.MoveToAccelerator();
  e.MoveToAccelerator();
  ls.MoveToAccelerator();

  mat.ConvertToELL();
  mat.info();

  double time = paralution_time();

  ls.Solve(b, &x);

  if (rank == 0) {
    std::cout << "Solve took: " << (paralution_time()-time)/1e6 << " sec" << std::endl;
  }

  e.ScaleAdd(-1.0, x);
  ValueType error = e.Norm();

  if (rank == 0) {
    std::cout << "||e - x||_2 = " << error << "\n";
  }

  stop_paralution();

  MPI_Finalize();

  return 0;

}
