#ifndef PARALUTION_OCL_KERNELS_COO_HPP_
#define PARALUTION_OCL_KERNELS_COO_HPP_

namespace paralution {

const char *ocl_kernels_coo = CL_KERNEL(

__kernel void kernel_coo_permute(         const IndexType  nnz,
                                 __global const IndexType *in_row,
                                 __global const IndexType *in_col,
                                 __global const IndexType *perm,
                                 __global       IndexType *out_row,
                                 __global       IndexType *out_col) {

  IndexType ind = get_global_id(0);

  for (IndexType i=ind; i<nnz; i+=get_local_size(0)) {

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
inline void segreduce_block(__local const IndexType *idx, __local ValueType *val) {

  ValueType left = ocl_set((ValueType) 0);
  IndexType tid = get_local_id(0);

  if( tid >=   1 && idx[tid] == idx[tid -   1] ) { left = val[tid -   1]; } barrier(CLK_LOCAL_MEM_FENCE); val[tid] += left; left = 0; barrier(CLK_LOCAL_MEM_FENCE);  
  if( tid >=   2 && idx[tid] == idx[tid -   2] ) { left = val[tid -   2]; } barrier(CLK_LOCAL_MEM_FENCE); val[tid] += left; left = 0; barrier(CLK_LOCAL_MEM_FENCE);
  if( tid >=   4 && idx[tid] == idx[tid -   4] ) { left = val[tid -   4]; } barrier(CLK_LOCAL_MEM_FENCE); val[tid] += left; left = 0; barrier(CLK_LOCAL_MEM_FENCE);
  if( tid >=   8 && idx[tid] == idx[tid -   8] ) { left = val[tid -   8]; } barrier(CLK_LOCAL_MEM_FENCE); val[tid] += left; left = 0; barrier(CLK_LOCAL_MEM_FENCE);
  if( tid >=  16 && idx[tid] == idx[tid -  16] ) { left = val[tid -  16]; } barrier(CLK_LOCAL_MEM_FENCE); val[tid] += left; left = 0; barrier(CLK_LOCAL_MEM_FENCE);
  if( tid >=  32 && idx[tid] == idx[tid -  32] ) { left = val[tid -  32]; } barrier(CLK_LOCAL_MEM_FENCE); val[tid] += left; left = 0; barrier(CLK_LOCAL_MEM_FENCE);  
  if( tid >=  64 && idx[tid] == idx[tid -  64] ) { left = val[tid -  64]; } barrier(CLK_LOCAL_MEM_FENCE); val[tid] += left; left = 0; barrier(CLK_LOCAL_MEM_FENCE);
  if( tid >= 128 && idx[tid] == idx[tid - 128] ) { left = val[tid - 128]; } barrier(CLK_LOCAL_MEM_FENCE); val[tid] += left; left = 0; barrier(CLK_LOCAL_MEM_FENCE);
  if( tid >= 256 && idx[tid] == idx[tid - 256] ) { left = val[tid - 256]; } barrier(CLK_LOCAL_MEM_FENCE); val[tid] += left; left = 0; barrier(CLK_LOCAL_MEM_FENCE);

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
__kernel void kernel_coo_spmv_flat(         const IndexType  num_nonzeros,
                                            const IndexType  interval_size,
                                   __global const IndexType *I, 
                                   __global const IndexType *J, 
                                   __global const ValueType *V, 
                                            const ValueType  scalar,
                                   __global const ValueType *x, 
                                   __global       ValueType *y,
                                   __global       IndexType *temp_rows,
                                   __global       ValueType *temp_vals) {

  __local volatile IndexType rows[48 * (BLOCK_SIZE/32) + WARP_SIZE/2];
  __local volatile ValueType vals[BLOCK_SIZE];

        IndexType tid         = get_local_id(0);
  const IndexType thread_id   = BLOCK_SIZE * get_group_id(0) + tid;
  const IndexType thread_lane = tid & (WARP_SIZE-1);
  const IndexType warp_id     = thread_id / WARP_SIZE;

  const IndexType interval_begin = warp_id * interval_size;
  IndexType interval_end2 = interval_begin + interval_size;
  if (interval_end2 > num_nonzeros)
    interval_end2 = num_nonzeros;

  const IndexType interval_end = interval_end2;

  const IndexType idx = WARP_SIZE/2 * (tid/WARP_SIZE + 1) + tid;

  rows[idx - WARP_SIZE/2] = -1;

  if(interval_begin >= interval_end)
    return;

  if (thread_lane == WARP_SIZE-1) {
    rows[idx] = I[interval_begin]; 
    vals[tid] = ocl_set((ValueType) 0);
  }

  for(IndexType n = interval_begin + thread_lane; n < interval_end; n += WARP_SIZE) {
    IndexType row = I[n];
    ValueType val = ocl_mult(ocl_mult(scalar, V[n]), x[J[n]]);
    
    if (thread_lane == 0) {
      if(row == rows[idx + WARP_SIZE-1])
        val += vals[tid + WARP_SIZE-1];
      else
        y[rows[idx + WARP_SIZE-1]] += vals[tid + WARP_SIZE-1];
    }

    rows[idx] = row;
    vals[tid] = val;

    if (row == rows[idx -  1]) { vals[tid] = val = val + vals[tid -  1]; } 
    if (row == rows[idx -  2]) { vals[tid] = val = val + vals[tid -  2]; }
    if (row == rows[idx -  4]) { vals[tid] = val = val + vals[tid -  4]; }
    if (row == rows[idx -  8]) { vals[tid] = val = val + vals[tid -  8]; }
    if (row == rows[idx - 16]) { vals[tid] = val = val + vals[tid - 16]; }
    if (WARP_SIZE > 32) if (row == rows[idx - 32]) { vals[tid] = val = val + vals[tid - 32]; }

    if(thread_lane < WARP_SIZE-1 && row != rows[idx + 1])
      y[row] += vals[tid];
  }

  if (thread_lane == WARP_SIZE-1) {
    temp_rows[warp_id] = rows[idx];
    temp_vals[warp_id] = vals[tid];
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
__kernel void kernel_coo_spmv_reduce_update(         const IndexType  num_warps,
                                            __global const IndexType *temp_rows,
                                            __global const ValueType *temp_vals,
                                            __global       ValueType *y) {

  __local IndexType rows[BLOCK_SIZE + 1];
  __local ValueType vals[BLOCK_SIZE + 1];

  IndexType tid = get_local_id(0);

  const IndexType end = num_warps - (num_warps & (BLOCK_SIZE - 1));

  if (tid == 0) {
    rows[BLOCK_SIZE] = (int) -1;
    vals[BLOCK_SIZE] = ocl_set((ValueType) 0);
  }
  
  barrier(CLK_LOCAL_MEM_FENCE);

  IndexType i = tid;

  while (i < end) {
    rows[tid] = temp_rows[i];
    vals[tid] = temp_vals[i];

    barrier(CLK_LOCAL_MEM_FENCE);

    segreduce_block(rows, vals);

    if (rows[tid] != rows[tid + 1])
      y[rows[tid]] += vals[tid];

    barrier(CLK_LOCAL_MEM_FENCE);

    i += BLOCK_SIZE; 
  }

  if (end < num_warps) {
    if (i < num_warps) {
      rows[tid] = temp_rows[i];
      vals[tid] = temp_vals[i];
    } else {
      rows[tid] = (int) -1;
      vals[tid] = ocl_set((ValueType) 0);
    }

    barrier(CLK_LOCAL_MEM_FENCE);
 
    segreduce_block(rows, vals);

    if (i < num_warps)
      if (rows[tid] != rows[tid + 1])
        y[rows[tid]] += vals[tid];
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
__kernel void kernel_coo_spmv_serial(         const IndexType  num_entries,
                                     __global const IndexType *I, 
                                     __global const IndexType *J, 
                                     __global const ValueType *V, 
                                              const ValueType  scalar,
                                     __global const ValueType *x, 
                                     __global       ValueType *y,
                                              const IndexType  shift) {

  for(IndexType n = 0; n < num_entries-shift; n++)
    y[I[n+shift]] += ocl_mult(ocl_mult(scalar, V[n+shift]), x[J[n+shift]]);

}

__kernel void kernel_coo_csr_to_coo(const IndexType nrow, __global const IndexType *row_offset, __global IndexType *row) {

  IndexType ai = get_global_id(0);
  IndexType aj;

  if (ai < nrow) {

    for (aj=row_offset[ai]; aj<row_offset[ai+1]; ++aj)
      row[aj] = ai;

  }

}

__kernel void kernel_coo_add_spmv(const IndexType nnz, __global const IndexType *row, __global const IndexType *col,
                                  __global const ValueType *val, const ValueType scalar, __global const ValueType *in,
                                  __global ValueType *out) {

  IndexType gid = get_global_id(0);

  if (gid < nnz) {

    ocl_atomic_add(&out[row[gid]], scalar * val[gid] * in[col[gid]]);

  }

}

);

}

#endif // PARALUTION_OCL_KERNELS_COO_HPP_
