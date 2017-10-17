#ifndef PARALUTION_OCL_KERNELS_VECTOR_HPP_
#define PARALUTION_OCL_KERNELS_VECTOR_HPP_

namespace paralution {

const char *ocl_kernels_vector = CL_KERNEL(

__kernel void kernel_scale(const IndexType size, const ValueType alpha, __global ValueType *x) {

  IndexType gid = get_global_id(0);

  if (gid < size)
    x[gid] = ocl_mult(alpha, x[gid]);

}

__kernel void kernel_scaleadd(         const IndexType size,
                                       const ValueType alpha,
                              __global const ValueType *x,
                              __global       ValueType *out) {

  IndexType gid = get_global_id(0);

  if (gid < size)
    out[gid] = ocl_mult(alpha, out[gid]) + x[gid];

}

__kernel void kernel_scaleaddscale(         const IndexType size,
                                            const ValueType alpha,
                                            const ValueType beta, 
                                   __global const ValueType *x,
                                   __global       ValueType *out) {

  IndexType gid = get_global_id(0);

  if (gid < size)
    out[gid] = ocl_mult(alpha, out[gid]) + ocl_mult(beta, x[gid]);

}

__kernel void kernel_scaleaddscale_offset(         const IndexType size,
                                                   const IndexType src_offset,
                                                   const IndexType dst_offset, 
                                                   const ValueType alpha,
                                                   const ValueType beta, 
                                          __global const ValueType *x,
                                          __global       ValueType *out) {

  IndexType gid = get_global_id(0);

  if (gid < size)
    out[gid+dst_offset] = ocl_mult(alpha, out[gid+dst_offset]) + ocl_mult(beta, x[gid+src_offset]);

}

__kernel void kernel_scaleadd2(         const IndexType size,
                                        const ValueType alpha,
                                        const ValueType beta,
                                        const ValueType gamma,
                               __global const ValueType *x,
                               __global const ValueType *y,
                               __global       ValueType *out) {

  IndexType gid = get_global_id(0);

  if (gid < size)
    out[gid] = ocl_mult(alpha, out[gid]) + ocl_mult(beta, x[gid]) + ocl_mult(gamma, y[gid]);

}

__kernel void kernel_pointwisemult(         const IndexType size,
                                   __global const ValueType *x,
                                   __global       ValueType *out) {

  IndexType gid = get_global_id(0);

  if (gid < size)
    out[gid] = ocl_mult(out[gid], x[gid]);

}

__kernel void kernel_pointwisemult2(         const IndexType size,
                                    __global const ValueType *x,
                                    __global const ValueType *y,
                                    __global       ValueType *out) {

  IndexType gid = get_global_id(0);

  if (gid < size)
    out[gid] = ocl_mult(y[gid], x[gid]);

}

__kernel void kernel_copy_offset_from(         const IndexType size,
                                               const IndexType src_offset,
                                               const IndexType dst_offset,
                                      __global const ValueType *in,
                                      __global       ValueType *out) {

  IndexType gid = get_global_id(0);

  if (gid < size)
    out[gid+dst_offset] = in[gid+src_offset];

}

__kernel void kernel_permute(         const IndexType size,
                             __global const IndexType *permute,
                             __global const ValueType *in,
                             __global       ValueType *out) {

  IndexType gid = get_global_id(0);

  if (gid < size)
    out[permute[gid]] = in[gid];

}

__kernel void kernel_permute_backward(         const IndexType size,
                                      __global const IndexType *permute,
                                      __global const ValueType *in,
                                      __global       ValueType *out) {

  IndexType gid = get_global_id(0);

  if (gid < size)
    out[gid] = in[permute[gid]];

}

__kernel void kernel_dot(         const IndexType  size,
                         __global const ValueType *x,
                         __global const ValueType *y,
                         __global       ValueType *out) {

  IndexType gid = get_global_id(0);
  IndexType tid = get_local_id(0);
  IndexType offset = WARP_SIZE * BLOCK_SIZE;

  __local ValueType sdata[BLOCK_SIZE];

  ValueType sum = ocl_set((ValueType) 0);

  for (IndexType i=gid; i<size; i+=offset) {
    sum = sum + ocl_mult(x[i], y[i]);
  }

  sdata[tid] = sum;

  barrier(CLK_LOCAL_MEM_FENCE);

  for (IndexType i=BLOCK_SIZE/2; i>0; i/=2) {

    if (tid < i) {
      sdata[tid] += sdata[tid + i];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

  }

  if (tid == 0) {
    out[get_group_id(0)] = sdata[tid];
  }

}

__kernel void kernel_dotc(         const IndexType size,
                          __global const ValueType *x,
                          __global const ValueType *y,
                          __global       ValueType *out) {

  IndexType gid = get_global_id(0);
  IndexType tid = get_local_id(0);
  IndexType offset = WARP_SIZE * BLOCK_SIZE;

  __local ValueType sdata[BLOCK_SIZE];

  ValueType sum = ocl_set((ValueType) 0);

  for (IndexType i=gid; i<size; i+=offset) {
    sum = sum + ocl_multc(x[i], y[i]);
  }

  sdata[tid] = sum;

  barrier(CLK_LOCAL_MEM_FENCE);

  for (IndexType i=BLOCK_SIZE/2; i>0; i/=2) {

    if (tid < i) {
      sdata[tid] += sdata[tid + i];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

  }

  if (tid == 0) {
    out[get_group_id(0)] = sdata[tid];
  }

}

__kernel void kernel_norm(        const IndexType size,
                         __global const ValueType *x,
                         __global       ValueType *out) {

  IndexType gid = get_global_id(0);
  IndexType tid = get_local_id(0);
  IndexType offset = WARP_SIZE * BLOCK_SIZE;

  __local ValueType sdata[BLOCK_SIZE];

  ValueType sum = ocl_set((ValueType) 0);

  for (IndexType i=gid; i<size; i+=offset) {
    sum = sum + ocl_norm(x[i]);
  }

  sdata[tid] = sum;

  barrier(CLK_LOCAL_MEM_FENCE);

  for (IndexType i=BLOCK_SIZE/2; i>0; i/=2) {

    if (tid < i) {
      sdata[tid] += sdata[tid + i];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

  }

  if (tid == 0) {
    out[get_group_id(0)] = sdata[tid];
  }

}

__kernel void kernel_axpy(         const IndexType size,
                                   const ValueType alpha,
                          __global const ValueType *x,
                          __global       ValueType *out) {

  IndexType gid = get_global_id(0);

  if (gid < size)
    out[gid] += ocl_mult(alpha, x[gid]);

}

__kernel void kernel_reduce(         const IndexType size,
                            __global const ValueType *data,
                            __global       ValueType *out) {

  IndexType gid = get_global_id(0);
  IndexType tid = get_local_id(0);
  IndexType offset = WARP_SIZE * BLOCK_SIZE;

  __local ValueType sdata[BLOCK_SIZE];

  ValueType sum = ocl_set((ValueType) 0);

  for (IndexType i=gid; i<size; i+=offset) {
    sum = sum + data[i];
  }

  sdata[tid] = sum;

  barrier(CLK_LOCAL_MEM_FENCE);

  for (IndexType i=BLOCK_SIZE/2; i>0; i/=2) {

    if (tid < i) {
      sdata[tid] += sdata[tid + i];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

  }

  if (tid == 0) {
    out[get_group_id(0)] = sdata[tid];
  }

}

__kernel void kernel_asum(         const IndexType size,
                          __global const ValueType *data,
                          __global       ValueType *out) {

  IndexType gid = get_global_id(0);
  IndexType tid = get_local_id(0);
  IndexType offset = WARP_SIZE * BLOCK_SIZE;

  __local ValueType sdata[BLOCK_SIZE];

  ValueType sum = ocl_set((ValueType) 0);

  for (IndexType i=gid; i<size; i+=offset) {
    sum = sum + ocl_abs(data[i]);
  }

  sdata[tid] = sum;

  barrier(CLK_LOCAL_MEM_FENCE);

  for (IndexType i=BLOCK_SIZE/2; i>0; i/=2) {

    if (tid < i) {
      sdata[tid] += sdata[tid + i];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

  }

  if (tid == 0) {
    out[get_group_id(0)] = sdata[tid];
  }

}

__kernel void kernel_get_index_values(         const IndexType size,
                                      __global const IndexType *index,
                                      __global const ValueType *in,
                                      __global       ValueType *out) {

  IndexType i = get_global_id(0);

  if (i < size)
    out[i] = in[index[i]];

}

__kernel void kernel_set_index_values(         const IndexType size,
                                      __global const IndexType *index,
                                      __global const ValueType *in,
                                      __global       ValueType *out) {

  IndexType i = get_global_id(0);

  if (i < size)
    out[index[i]] = in[i];

}

__kernel void kernel_amax(         const IndexType size,
                          __global const ValueType *data,
                          __global       ValueType *out,
                          __global       IndexType *iout) {

  IndexType gid = get_global_id(0);
  IndexType tid = get_local_id(0);
  IndexType offset = WARP_SIZE * BLOCK_SIZE;

  __local IndexType idata[BLOCK_SIZE];
  __local ValueType sdata[BLOCK_SIZE];

  IndexType imax = 0;
  ValueType smax = ocl_set((ValueType) 0);

  for (IndexType i=gid; i<size; i+=offset) {
    ValueType tmp = data[i];
    if (ocl_abs(tmp) > ocl_abs(smax)) {
      smax = ocl_abs(tmp);
      imax = i;
    }
  }

  idata[tid] = imax;
  sdata[tid] = smax;

  barrier(CLK_LOCAL_MEM_FENCE);

  for (IndexType i=BLOCK_SIZE/2; i>0; i/=2) {

    if (tid < i) {
      ValueType tmp = sdata[tid+i];
      if (ocl_abs(tmp) > ocl_abs(sdata[tid])) {
        sdata[tid] = ocl_abs(tmp);
        idata[tid] = idata[tid+i];
      }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

  }

  if (tid == 0) {
    out[get_group_id(0)] = sdata[tid];
    iout[get_group_id(0)] = idata[tid];
  }

}

__kernel void kernel_power(const IndexType n, const double power, __global ValueType *out) {

  IndexType gid = get_global_id(0);

  if (gid < n)
    out[gid] = ocl_pow(out[gid], (ValueType) power);

}

__kernel void kernel_copy_from_float(const IndexType n, __global const float *in, __global ValueType *out) {

  IndexType ind = get_global_id(0);

  if (ind < n)
    out[ind] = (ValueType) in[ind];

}

__kernel void kernel_copy_from_double(const IndexType n, __global const double *in, __global ValueType *out) {

  IndexType ind = get_global_id(0);

  if (ind < n)
    out[ind] = (ValueType) in[ind];

}

);

}

#endif // PARALUTION_OCL_KERNELS_VECTOR_HPP_
