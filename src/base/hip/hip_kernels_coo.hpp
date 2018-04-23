#ifndef PARALUTION_HIP_HIP_KERNELS_COO_HPP_
#define PARALUTION_HIP_HIP_KERNELS_COO_HPP_

#include "../matrix_formats_ind.hpp"

namespace paralution {

template <typename ValueType, typename IndexType>
__global__ void kernel_coo_permute(const IndexType nnz, 
                                   const IndexType *in_row, const IndexType *in_col, 
                                   const IndexType *perm,
                                   IndexType *out_row,  IndexType *out_col) {


  IndexType ind = blockIdx.x*blockDim.x+threadIdx.x;

  for (int i=ind; i<nnz; i+=gridDim.x) {

    out_row[i] = perm[ in_row[i] ];
    out_col[i] = perm[ in_col[i] ];

  }

}

// ----------------------------------------------------------
// function segreduce_block(...)
// ----------------------------------------------------------
// Modified and adapted from CUSP 0.3.1, 
// http://code.google.com/p/cusp-library/
// NVIDIA, APACHE LICENSE 2.0
// ----------------------------------------------------------
// CHANGELOG
// - adapted interface
// ----------------------------------------------------------  
template <typename IndexType, typename ValueType>
__device__ void segreduce_block(const IndexType * idx, ValueType * val)
{
  ValueType left;
  make_ValueType(left, 0.0);

  if( threadIdx.x >=   1 && idx[threadIdx.x] == idx[threadIdx.x -   1] ) { left = val[threadIdx.x -   1]; } __syncthreads(); val[threadIdx.x] = val[threadIdx.x] + left; make_ValueType(left, 0.0); __syncthreads();  
  if( threadIdx.x >=   2 && idx[threadIdx.x] == idx[threadIdx.x -   2] ) { left = val[threadIdx.x -   2]; } __syncthreads(); val[threadIdx.x] = val[threadIdx.x] + left; make_ValueType(left, 0.0); __syncthreads();
  if( threadIdx.x >=   4 && idx[threadIdx.x] == idx[threadIdx.x -   4] ) { left = val[threadIdx.x -   4]; } __syncthreads(); val[threadIdx.x] = val[threadIdx.x] + left; make_ValueType(left, 0.0); __syncthreads();
  if( threadIdx.x >=   8 && idx[threadIdx.x] == idx[threadIdx.x -   8] ) { left = val[threadIdx.x -   8]; } __syncthreads(); val[threadIdx.x] = val[threadIdx.x] + left; make_ValueType(left, 0.0); __syncthreads();
  if( threadIdx.x >=  16 && idx[threadIdx.x] == idx[threadIdx.x -  16] ) { left = val[threadIdx.x -  16]; } __syncthreads(); val[threadIdx.x] = val[threadIdx.x] + left; make_ValueType(left, 0.0); __syncthreads();
  if( threadIdx.x >=  32 && idx[threadIdx.x] == idx[threadIdx.x -  32] ) { left = val[threadIdx.x -  32]; } __syncthreads(); val[threadIdx.x] = val[threadIdx.x] + left; make_ValueType(left, 0.0); __syncthreads();  
  if( threadIdx.x >=  64 && idx[threadIdx.x] == idx[threadIdx.x -  64] ) { left = val[threadIdx.x -  64]; } __syncthreads(); val[threadIdx.x] = val[threadIdx.x] + left; make_ValueType(left, 0.0); __syncthreads();
  if( threadIdx.x >= 128 && idx[threadIdx.x] == idx[threadIdx.x - 128] ) { left = val[threadIdx.x - 128]; } __syncthreads(); val[threadIdx.x] = val[threadIdx.x] + left; make_ValueType(left, 0.0); __syncthreads();
  if( threadIdx.x >= 256 && idx[threadIdx.x] == idx[threadIdx.x - 256] ) { left = val[threadIdx.x - 256]; } __syncthreads(); val[threadIdx.x] = val[threadIdx.x] + left; make_ValueType(left, 0.0); __syncthreads();
}


// ----------------------------------------------------------
// function kernel_spmv_coo_flat(...)
// ----------------------------------------------------------
// Modified and adapted from CUSP 0.3.1, 
// http://code.google.com/p/cusp-library/
// NVIDIA, APACHE LICENSE 2.0
// ----------------------------------------------------------
// CHANGELOG
// - adapted interface
// ----------------------------------------------------------  
template <unsigned int BLOCK_SIZE, unsigned int WARP_SIZE, typename IndexType, typename ValueType>
__launch_bounds__(BLOCK_SIZE,1)
__global__ void kernel_spmv_coo_flat(const IndexType nnz,
                                     const IndexType interval_size,
                                     const IndexType *I,
                                     const IndexType *J,
                                     const ValueType *V,
                                     const ValueType scalar,
                                     const ValueType *x,
                                           ValueType *y,
                                           IndexType *temp_rows,
                                           ValueType *temp_vals) {

  __shared__ volatile IndexType rows[48 *(BLOCK_SIZE/32)];
  __shared__ volatile ValueType vals[BLOCK_SIZE];

  // global thread index
  const IndexType thread_id   = BLOCK_SIZE * blockIdx.x + threadIdx.x;
  // thread index within the warp
  const IndexType thread_lane = threadIdx.x & (WARP_SIZE-1);
  // global warp index
  const IndexType warp_id     = thread_id   / WARP_SIZE;
  // warp's offset into I,J,V
  const IndexType interval_begin = warp_id * interval_size;
  // end of warps's work
  const IndexType interval_end   = (interval_begin+interval_size < nnz) ? interval_begin+interval_size : nnz;
  // thread's index into padded rows array
  const IndexType idx = 16 * (threadIdx.x/32 + 1) + threadIdx.x;
  // fill padding with invalid row index
  rows[idx - 16] = -1;

  // warp has no work to do
  if(interval_begin >= interval_end) {
    return;
  }

  ValueType zero;
  ValueType one;
  make_ValueType(zero, 0.0);
  make_ValueType(one, 1.0);

  // initialize the carry in values
  if (thread_lane == 31) {
    rows[idx] = I[interval_begin];
    assign_volatile_ValueType(&zero, &vals[threadIdx.x]);
  }

  for(IndexType n=interval_begin+thread_lane; n<interval_end; n+=WARP_SIZE) {

    // row index (i)
    IndexType row = I[n];
    // A(i,j) * x(j)
    ValueType val = scalar * V[n] * x[J[n]];

    if (thread_lane == 0) {
      if(row == rows[idx + 31]) {
        // row continues
        val = add_volatile_ValueType(&vals[threadIdx.x+31], &val);
      } else {
        // row terminated
        y[rows[idx+31]] = add_volatile_ValueType(&vals[threadIdx.x+31], &y[rows[idx+31]]);
      }
    }

    rows[idx]         = row;
    assign_volatile_ValueType(&val, &vals[threadIdx.x]);

    if(row == rows[idx -  1]) {
      val = add_volatile_ValueType(&vals[threadIdx.x -  1], &val);
      assign_volatile_ValueType(&val, &vals[threadIdx.x]);
    }
    if(row == rows[idx -  2]) {
      val = add_volatile_ValueType(&vals[threadIdx.x -  2], &val);
      assign_volatile_ValueType(&val, &vals[threadIdx.x]);
    }
    if(row == rows[idx -  4]) {
      val = add_volatile_ValueType(&vals[threadIdx.x -  4], &val);
      assign_volatile_ValueType(&val, &vals[threadIdx.x]);
    }
    if(row == rows[idx -  8]) {
      val = add_volatile_ValueType(&vals[threadIdx.x -  8], &val);
      assign_volatile_ValueType(&val, &vals[threadIdx.x]);
    }
    if(row == rows[idx - 16]) {
      val = add_volatile_ValueType(&vals[threadIdx.x - 16], &val);
      assign_volatile_ValueType(&val, &vals[threadIdx.x]);
    }

    // row terminated
    if(thread_lane < 31 && row != rows[idx + 1]) {
      y[row] = add_volatile_ValueType(&vals[threadIdx.x], &y[row]);
    }

  }

  // write the carry out values
  if(thread_lane == 31) {
    temp_rows[warp_id] = rows[idx];
    assign_volatile_ValueType(&vals[threadIdx.x], &temp_vals[warp_id]);
  }

}

// ----------------------------------------------------------
// function kernel_spmv_coo_reduce_update(...)
// ----------------------------------------------------------
// Modified and adapted from CUSP 0.3.1, 
// http://code.google.com/p/cusp-library/
// NVIDIA, APACHE LICENSE 2.0
// ----------------------------------------------------------
// CHANGELOG
// - adapted interface
// ----------------------------------------------------------  
template <unsigned int BLOCK_SIZE, typename IndexType, typename ValueType>
__launch_bounds__(BLOCK_SIZE, 1)
__global__ void kernel_spmv_coo_reduce_update(const IndexType num_warps,
                                              const IndexType * temp_rows,
                                              const ValueType * temp_vals,
                                              ValueType * y) {

  __shared__ IndexType rows[BLOCK_SIZE + 1];    
  __shared__ ValueType vals[BLOCK_SIZE + 1];    

  const IndexType end = num_warps - (num_warps & (BLOCK_SIZE - 1));

  if (threadIdx.x == 0) {
    rows[BLOCK_SIZE] = (IndexType) -1;
    make_ValueType(vals[BLOCK_SIZE], 0.0);
  }
  
  __syncthreads();

  IndexType i = threadIdx.x;

  while (i < end) {

    // do full blocks
    rows[threadIdx.x] = temp_rows[i];
    vals[threadIdx.x] = temp_vals[i];

    __syncthreads();

    segreduce_block(rows, vals);

    if (rows[threadIdx.x] != rows[threadIdx.x + 1]) {
      y[rows[threadIdx.x]] = y[rows[threadIdx.x]] + vals[threadIdx.x];
    }

    __syncthreads();

    i += BLOCK_SIZE;

  }

  if (end < num_warps) {

    if (i < num_warps) {

      rows[threadIdx.x] = temp_rows[i];
      vals[threadIdx.x] = temp_vals[i];

    } else {

      rows[threadIdx.x] = (IndexType) -1;
      make_ValueType(vals[threadIdx.x], 0.0);

    }

    __syncthreads();
 
    segreduce_block(rows, vals);

    if (i < num_warps) {

      if (rows[threadIdx.x] != rows[threadIdx.x + 1]) {

        y[rows[threadIdx.x]] = y[rows[threadIdx.x]] + vals[threadIdx.x];

      }

    }

  }

}

// ----------------------------------------------------------
// function spmv_coo_serial_kernel(...)
// ----------------------------------------------------------
// Modified and adapted from CUSP 0.3.1, 
// http://code.google.com/p/cusp-library/
// NVIDIA, APACHE LICENSE 2.0
// ----------------------------------------------------------
// CHANGELOG
// - adapted interface
// ----------------------------------------------------------  
template <typename IndexType, typename ValueType>
__global__ void kernel_spmv_coo_serial(const IndexType nnz, const IndexType *row, const IndexType *col,
                                       const ValueType *val, const ValueType scalar, const ValueType *in,
                                       ValueType *out) {

  for (IndexType n=0; n<nnz; ++n) {
    out[row[n]] = out[row[n]] + scalar * val[n] * in[col[n]];
  }

}

template <typename IndexType, typename ValueType>
__global__ void kernel_spmv_add_coo(const IndexType nnz, const IndexType *row, const IndexType *col, const ValueType *val,
                                    const ValueType *in, const ValueType alpha, ValueType *out) {

  int idx = blockDim.x * blockIdx.x + threadIdx.x;

  if (idx < nnz) {

    atomic_add_hip(&out[row[idx]], alpha * val[idx] * in[col[idx]]);

  }

}

template <typename IndexType>
__global__ void kernel_coo_csr_to_coo(const IndexType nrow, const IndexType *row_offset, IndexType *row) {

  IndexType ai = blockIdx.x * blockDim.x + threadIdx.x;
  IndexType aj;

  if (ai < nrow) {

    for (aj=row_offset[ai]; aj<row_offset[ai+1]; ++aj)
      row[aj] = ai;

  }

}

}

#endif // PARALUTION_HIP_HIP_KERNELS_COO_HPP_
