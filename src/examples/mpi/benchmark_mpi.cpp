#include <iostream>
#include <fstream>
#include <algorithm>
#include <vector>
#include <map>
#include <mpi.h>
#include <rocalution.hpp>

#define ValueType double

using namespace rocalution;

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

  if (argc < 2) { 
    std::cerr << argv[0] << " <global_matrix>" << std::endl;
    return -1;
  }

  // Initialize platform with rank and # of accelerator devices in the node
  set_omp_affinity(false);

  init_rocalution(rank, 1);

  // Disable OpenMP
  set_omp_threads_rocalution(1);

  // Print platform
  info_rocalution();

  // Load undistributed matrix
  LocalMatrix<ValueType> undis_mat;
  undis_mat.ReadFileCSR(argv[1]);

  size_t global_nrow = undis_mat.get_nrow();

  int *global_row_offset = NULL;
  int *global_col = NULL;
  ValueType *global_val = NULL;

  undis_mat.LeaveDataPtrCSR(&global_row_offset, &global_col, &global_val);

  // Compute local matrix sizes
  std::vector<int> local_size(num_procs);

  for (int i=0; i<num_procs; ++i) {
    local_size[i] = global_nrow / num_procs;
  }

  if (global_nrow % num_procs != 0) {
    for (size_t i=0; i<global_nrow % num_procs; ++i) {
      ++local_size[i];
    }
  }

  // Compute index offsets
  std::vector<int> index_offset(num_procs+1);
  index_offset[0] = 0;
  for (int i=0; i<num_procs; ++i) {
    index_offset[i+1] = index_offset[i] + local_size[i];
  }

  // Read sub matrix - row_offset
  int local_nrow = local_size[rank];
  std::vector<int> local_row_offset(local_nrow+1);

  for (int i=index_offset[rank], k=0; k<local_nrow+1; ++i, ++k) {
    local_row_offset[k] = global_row_offset[i];
  }

  // Read sub matrix - col and val
  int local_nnz = local_row_offset[local_nrow] - local_row_offset[0];
  std::vector<int> local_col(local_nnz);
  std::vector<ValueType> local_val(local_nnz);

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
  std::vector<int> boundary_size(neighbors);

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
  std::vector<int> recv_offset(neighbors+1);
  std::vector<int> send_offset(neighbors+1);
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
  std::vector<std::vector<int> > local_boundary(neighbors);
  for (int i=0; i<neighbors; ++i) {
    local_boundary[i].resize(boundary_size[i]);
  }

  // MPI receive boundary
  for (int i=0; i<num_procs; ++i) {
    // If neighbor receive from rank i is expected...
    if (neighbor[i] == true) {
      // Receive boundary from rank i to current rank
      MPI_Irecv(local_boundary[n].data(), boundary_size[n], MPI_INT, i, 0, comm, &mpi_req[n]);
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
  std::vector<int> bnd(boundary_nnz);

  for (int i=0; i<num_procs; ++i) {
    for (unsigned int j=0; j<boundary[i].size(); ++j) {
      bnd[k] = boundary[i][j]-index_offset[rank];
      ++k;
    }
  }

  // Create boundary index array
  std::vector<int> boundary_index(nnz_boundary);

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

  std::vector<int> recv(neighbors);
  std::vector<int> sender(neighbors);

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
  manager.SetBoundaryIndex(boundary_nnz, bnd.data());
  manager.SetReceivers(neighbors, recv.data(), recv_offset.data());
  manager.SetSenders(neighbors, sender.data(), send_offset.data());

  GlobalMatrix<ValueType> mat(manager);
  GlobalVector<ValueType> v1(manager);
  GlobalVector<ValueType> v2(manager);

  mat.SetLocalDataPtrCSR(&row_offset, &col, &val, "mat", interior_nnz);
  mat.SetGhostDataPtrCOO(&ghost_row, &ghost_col, &ghost_val, "ghost", ghost_nnz);

  mat.MoveToAccelerator();
  v1.MoveToAccelerator();
  v2.MoveToAccelerator();

  v1.Allocate("v1", mat.get_ncol());
  v2.Allocate("v2", mat.get_nrow());

  size_t size = mat.get_nrow();
  size_t nnz;

  v1.Ones();
  v2.Zeros();
  mat.Apply(v1, &v2);

  mat.info();
  v1.info();
  v2.info();

  const int tests = 200;

  double time;

  if (rank == 0) {
    std::cout << "--------------------------------------- BENCHMARKS --------------------------------------" << std::endl;
  }

  // Dot product
  // Size = 2*size
  // Flop = 2 per element
  v1.Dot(v2);

  _rocalution_sync();
  time = rocalution_time();

  for (int i=0; i<tests; ++i) {
    v1.Dot(v2);
    _rocalution_sync();
  }

  _rocalution_sync();
  time = rocalution_time() - time;

  if (rank == 0) {
    std::cout << "Dot execution: " << time/1e3/tests << " msec" << "; " <<
                 tests*double(sizeof(ValueType)*(2*size))/time/1e3 << " GByte/sec; " <<
                 tests*double(2*size)/time/1e3 << " GFlop/sec" << std::endl;
  }

  // L2 Norm
  // Size = size
  // Flop = 2 per element
  v1.Norm();

  _rocalution_sync();
  time = rocalution_time();

  for (int i=0; i<tests; ++i) {
    v1.Norm();
    _rocalution_sync();
  }

  _rocalution_sync();
  time = rocalution_time() - time;

  if (rank == 0) {
    std::cout << "Norm2 execution: " << time/1e3/tests << " msec" << "; "
              << tests*double(sizeof(ValueType)*(size))/time/1e3 << " GByte/sec; "
              << tests*double(2*size)/time/1e3 << " GFlop/sec" << std::endl;
  }

  // Reduction
  // Size = size
  // Flop = 1 per element
  v1.Reduce();

  _rocalution_sync();
  time = rocalution_time();

  for (int i=0; i<tests; ++i) {
    v1.Reduce();
    _rocalution_sync();
  }

  _rocalution_sync();
  time = rocalution_time() - time;

  if (rank == 0) {
    std::cout << "Reduce execution: " << time/1e3/tests << " msec" << "; "
              << tests*double(sizeof(ValueType)*(size))/time/1e3 << " GByte/sec; "
              << tests*double(size)/time/1e3 << " GFlop/sec" << std::endl;
  }

  if (rank == 0) {
    std::cout << "-----------------------------------------------------------------------------------------" << std::endl;
  }

  // Vector Update 1
  // Size = 3*size
  // Flop = 2 per element
  v1.ScaleAdd((ValueType) 3.1, v2);

  _rocalution_sync();
  time = rocalution_time();

  for (int i=0; i<tests; ++i) {
    v1.ScaleAdd((ValueType) 3.1, v2);
    _rocalution_sync();
  }

  _rocalution_sync();
  time = rocalution_time() - time;

  if (rank == 0) {
    std::cout << "Vector update (ScaleAdd) execution: " << time/1e3/tests << " msec" << "; "
              << tests*double(sizeof(ValueType)*(3*size))/time/1e3 << " GByte/sec; "
              << tests*double(2*size)/time/1e3 << " GFlop/sec" << std::endl;
  }

  // Vector Update 2
  // Size = 3*size
  // Flop = 2 per element
  v1.AddScale(v2, (ValueType) 3.1);
  _rocalution_sync();

  time = rocalution_time();

  for (int i=0; i<tests; ++i) {
    v1.AddScale(v2, (ValueType) 3.1);
    _rocalution_sync();
  }

  _rocalution_sync();
  time = rocalution_time() - time;

  if (rank == 0) {
    std::cout << "Vector update (AddScale) execution: " << time/1e3/tests << " msec" << "; "
              << tests*double(sizeof(ValueType)*(3*size))/time/1e3 << " GByte/sec; "
              << tests*double(2*size)/time/1e3 << " GFlop/sec" << std::endl;
  }

  if (rank == 0) {
    std::cout << "-----------------------------------------------------------------------------------------" << std::endl;
  }

  // Matrix vector multiplication CSR
  // Size = int(size+1+nnz) [row_offset + col] + ValueType(2*size+nnz) [in + out + nnz]
  // Flop = 2 per entry (nnz)

  mat.ConvertToCSR();
  nnz = mat.get_nnz();

  mat.info();

  mat.Apply(v1, &v2);

  _rocalution_sync();
  time = rocalution_time();

  for (int i=0; i<tests; ++i) {
    mat.Apply(v1, &v2);
    _rocalution_sync();
  }

  _rocalution_sync();
  time = rocalution_time() - time;

  if (rank == 0) {
    std::cout << "CSR SpMV execution: " << time/1e3/tests << " msec" << "; "
              << tests*double((sizeof(ValueType)*(2*size+nnz)+sizeof(int)*(size+1+nnz)))/time/1e3 << " GByte/sec; "
              << tests*double(2*nnz)/time/1e3 << " GFlop/sec" << std::endl;
  }

  if (rank == 0) {
    std::cout << "-----------------------------------------------------------------------------------------" << std::endl;
  }

  // Matrix vector multiplication MCSR
  // Size = int(size+(nnz-size)) [row_offset + col] + valuetype(2*size+nnz) [in + out + nnz]
  // Flop = 2 per entry (nnz)

  mat.ConvertToMCSR();
  nnz = mat.get_nnz();

  mat.info();

  mat.Apply(v1, &v2);

  _rocalution_sync();
  time = rocalution_time();

  for (int i=0; i<tests; ++i) {
    mat.Apply(v1, &v2);
    _rocalution_sync();
  }

  _rocalution_sync();
  time = rocalution_time() - time;

  if (rank == 0) {
    std::cout << "MCSR SpMV execution: " << time/1e3/tests << " msec" << "; "
              << tests*double((sizeof(ValueType)*(2*size+nnz-size)+sizeof(int)*(size+1+nnz)))/time/1e3 << " GByte/sec; "
              << tests*double(2*nnz)/time/1e3 << " GFlop/sec" << std::endl;
  }

  if (rank == 0) {
    std::cout << "-----------------------------------------------------------------------------------------" << std::endl;
  }

  // Matrix vector multiplication ELL
  // Size = int(nnz) [col] + ValueType(2*size+nnz) [in + out + nnz]
  // Flop = 2 per entry (nnz)

  mat.ConvertToELL();
  nnz = mat.get_nnz();

  mat.info();

  mat.Apply(v1, &v2);

  _rocalution_sync();
  time = rocalution_time();

  for (int i=0; i<tests; ++i) {
    mat.Apply(v1, &v2);
    _rocalution_sync();
  }

  _rocalution_sync();
  time = rocalution_time() - time;

  if (rank == 0) {
    std::cout << "ELL SpMV execution: " << time/1e3/tests << " msec" << "; "
              << tests*double((sizeof(ValueType)*(2*size+nnz)+sizeof(int)*(nnz)))/time/1e3 << " GByte/sec; "
              << tests*double(2*nnz)/time/1e3 << " GFlop/sec" << std::endl;
  }

  if (rank == 0) {
    std::cout << "-----------------------------------------------------------------------------------------" << std::endl;
  }

  // Matrix vector multiplication COO
  // Size = int(2*nnz) [col+row] + ValueType(2*size+nnz) [in + out + nnz]
  // Flop = 2 per entry (nnz)

  mat.ConvertToCOO();
  nnz = mat.get_nnz();

  mat.info();

  mat.Apply(v1, &v2);

  _rocalution_sync();
  time = rocalution_time();

  for (int i=0; i<tests; ++i) {
    mat.Apply(v1, &v2);
    _rocalution_sync();
  }

  _rocalution_sync();
  time = rocalution_time() - time;

  if (rank == 0) {
    std::cout << "COO SpMV execution: " << time/1e3/tests << " msec" << "; "
              << tests*double((sizeof(ValueType)*(2*size+nnz)+sizeof(int)*(2*nnz)))/time/1e3 << " GByte/sec; "
              << tests*double(2*nnz)/time/1e3 << " GFlop/sec" << std::endl;
  }

  if (rank == 0) {
    std::cout << "-----------------------------------------------------------------------------------------" << std::endl;
  }

  // Matrix vector multiplication HYB
  // Size = int(nnz) [col] + valuetype(2*size+nnz) [in + out + nnz]  
  // Flop = 2 per entry (nnz)

  mat.ConvertToHYB();
  nnz = mat.get_nnz();

  mat.info();

  mat.Apply(v1, &v2);

  _rocalution_sync();
  time = rocalution_time();

  for (int i=0; i<tests; ++i) {
    mat.Apply(v1, &v2);
    _rocalution_sync();
  }

  _rocalution_sync();
  time = rocalution_time() - time;

  if (rank == 0) {
    std::cout << "HYB SpMV execution: " << time/1e3/tests << " msec" << "; "
              << tests*double((sizeof(ValueType)*(2*size+nnz)+sizeof(int)*(nnz)))/time/1e3 << " GByte/sec; "
              << tests*double(2*nnz)/time/1e3 << " GFlop/sec" << std::endl;
  }

  if (rank == 0) {
    std::cout << "-----------------------------------------------------------------------------------------" << std::endl;
  }

  // Matrix vector multiplication DIA
  // Size = int(size+nnz) + valuetype(2*size+nnz)
  // Flop = 2 per entry (nnz)

  mat.ConvertToDIA();
  nnz = mat.get_nnz();

  mat.info();

  mat.Apply(v1, &v2);

  _rocalution_sync();
  time = rocalution_time();

  for (int i=0; i<tests; ++i) {
    mat.Apply(v1, &v2);
    _rocalution_sync();
  }

  _rocalution_sync();
  time = rocalution_time() - time;

  if (rank == 0) {
    std::cout << "DIA SpMV execution: " << time/1e3/tests << " msec" << "; "
              << tests*double((sizeof(ValueType)*(nnz)))/time/1e3 << " GByte/sec; "
              << tests*double(2*nnz)/time/1e3 << " GFlop/sec" << std::endl;
  }

  if (rank == 0) {
    std::cout << "-----------------------------------------------------------------------------------------" << std::endl;
  }

  mat.ConvertToCSR();

  if (rank == 0) {
    std::cout << "----------------------------------------------------" << std::endl;
    std::cout << "Combined micro benchmarks" << std::endl;
  }

  double dot_tick=0, dot_tack=0;
  double norm_tick=0, norm_tack=0;
  double red_tick=0, red_tack=0;
  double updatev1_tick=0, updatev1_tack=0;
  double updatev2_tick=0, updatev2_tack=0;
  double spmv_tick=0, spmv_tack=0;

  for (int i=0; i<tests; ++i) {

    v1.Ones();
    v2.Zeros();
    mat.Apply(v1, &v2);


    // Dot product
    // Size = 2*size
    // Flop = 2 per element
    v1.Dot(v2);
    
    dot_tick += rocalution_time();
    
      v1.Dot(v2);
    
    dot_tack += rocalution_time();
    

    v1.Ones();
    v2.Zeros();
    mat.Apply(v1, &v2);
    
    // Norm
    // Size = size
    // Flop = 2 per element
    v1.Norm();
    
    norm_tick += rocalution_time();
    
      v1.Norm();
    
    norm_tack += rocalution_time();
    
    
    v1.Ones();
    v2.Zeros();
    mat.Apply(v1, &v2);

    // Reduce
    // Size = size
    // Flop = 1 per element
    v1.Reduce();
    
    red_tick += rocalution_time();
    
      v1.Reduce();
    
    red_tack += rocalution_time();
    
    
    v1.Ones();
    v2.Zeros();
    mat.Apply(v1, &v2);

    // Vector Update 1
    // Size = 3xsize
    // Flop = 2 per element
    v1.ScaleAdd(double(5.5), v2);
    
    updatev1_tick += rocalution_time();
    
      v1.ScaleAdd(double(5.5), v2);
    
    updatev1_tack += rocalution_time(); 
    
    
    v1.Ones();
    v2.Zeros();
    mat.Apply(v1, &v2);
  
    // Vector Update 2
    // Size = 3*size
    // Flop = 2 per element
    v1.AddScale(v2, double(5.5));
    
    updatev2_tick += rocalution_time();
    
      v1.AddScale(v2, double(5.5));
    
    updatev2_tack += rocalution_time();
   
    v1.Ones();
    v2.Zeros();
    mat.Apply(v1, &v2);
    
    // Matrix-Vector Multiplication
    // Size = int(size+nnz) + valuetype(2*size+nnz)
    // Flop = 2 per entry (nnz)
    mat.Apply(v1, &v2);
    
    spmv_tick += rocalution_time();
    
      mat.Apply(v1, &v2);    
    
    spmv_tack += rocalution_time();

  }

  if (rank == 0) {
    std::cout << "Dot execution: " << (dot_tack-dot_tick)/tests/1e3 << " msec" << "; "
	      << tests*double(sizeof(double)*(size+size))/(dot_tack-dot_tick)/1e3 << " Gbyte/sec; "
        << tests*double(2*size)/(dot_tack-dot_tick)/1e3 << " GFlop/sec" << std::endl;

    std::cout << "Norm execution: " << (norm_tack-norm_tick)/tests/1e3 << " msec" << "; "
	      << tests*double(sizeof(double)*(size))/(norm_tack-norm_tick)/1e3 << " Gbyte/sec; "
	      << tests*double(2*size)/(norm_tack-norm_tick)/1e3 << " GFlop/sec" << std::endl;

    std::cout << "Reduce execution: " << (red_tack-red_tick)/tests/1e3 << " msec" << "; "
	      << tests*double(sizeof(double)*(size))/(red_tack-red_tick)/1e3 << " Gbyte/sec; "
	      << tests*double(size)/(red_tack-red_tick)/1e3 << " GFlop/sec" << std::endl;

    std::cout << "Vector update (scaleadd) execution: " << (updatev1_tack-updatev1_tick)/tests/1e3 << " msec" << "; "
	      << tests*double(sizeof(double)*(size+size+size))/(updatev1_tack-updatev1_tick)/1e3 << " Gbyte/sec; "
	      << tests*double(2*size)/(updatev1_tack-updatev1_tick)/1e3 << " GFlop/sec" << std::endl;

    std::cout << "Vector update (addscale) execution: " << (updatev2_tack-updatev2_tick)/tests/1e3 << " msec" << "; "
	      << tests*double(sizeof(double)*(size+size+size))/(updatev2_tack-updatev2_tick)/1e3 << " Gbyte/sec; "
	      << tests*double(2*size)/(updatev2_tack-updatev2_tick)/1e3 << " GFlop/sec" << std::endl;

    std::cout << "SpMV execution: " << (spmv_tack-spmv_tick)/tests/1e3 << " msec" << "; "
	      << tests*double((sizeof(double)*(size+size+nnz)+sizeof(int)*(size+nnz)))/(spmv_tack-spmv_tick)/1e3 << " Gbyte/sec; "
	      << tests*double((2*nnz)/(spmv_tack-spmv_tick))/1e3 << " GFlop/sec" << std::endl;
  }

  stop_rocalution();

  MPI_Finalize();

  return 0;

}
