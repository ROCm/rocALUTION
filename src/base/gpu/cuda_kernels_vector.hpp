#ifndef PARALUTION_GPU_CUDA_KERNELS_VECTOR_HPP_
#define PARALUTION_GPU_CUDA_KERNELS_VECTOR_HPP_

#include "gpu_complex.hpp"

namespace paralution {

template <typename ValueType, typename IndexType>
__global__ void kernel_scaleadd(const IndexType n, const ValueType alpha, const ValueType *x, ValueType *out) {

  IndexType ind = blockIdx.x * blockDim.x + threadIdx.x;

  if (ind < n)
    out[ind] = alpha * out[ind] + x[ind];

}

template <typename ValueType, typename IndexType>
__global__ void kernel_scaleaddscale(const IndexType n, const ValueType alpha, const ValueType beta,
                                     const ValueType *x, ValueType *out) {

  IndexType ind = blockIdx.x * blockDim.x + threadIdx.x;

  if (ind < n)
    out[ind] = alpha*out[ind] + beta*x[ind];

}

template <typename ValueType, typename IndexType>
__global__ void kernel_scaleaddscale_offset(const IndexType n,
                                            const IndexType src_offset, const IndexType dst_offset,
                                            const ValueType alpha, const ValueType beta,
                                            const ValueType *x, ValueType *out) {

  IndexType ind = blockIdx.x * blockDim.x + threadIdx.x;

  if (ind < n)
    out[ind+dst_offset] = alpha*out[ind+dst_offset] + beta*x[ind+src_offset];

}

template <typename ValueType, typename IndexType>
__global__ void kernel_scaleadd2(const IndexType n, const ValueType alpha, const ValueType beta, const ValueType gamma,
                                 const ValueType *x, const ValueType *y, ValueType *out) {

  IndexType ind = blockIdx.x * blockDim.x + threadIdx.x;

  if (ind < n)
    out[ind] = alpha*out[ind] + beta*x[ind] + gamma*y[ind];

}

template <typename ValueType, typename IndexType>
__global__ void kernel_pointwisemult(const IndexType n, const ValueType *x, ValueType *out) {

  IndexType ind = blockIdx.x * blockDim.x + threadIdx.x;

  if (ind < n)
    out[ind] = out[ind] * x[ind];

}

template <typename ValueType, typename IndexType>
__global__ void kernel_pointwisemult2(const IndexType n, const ValueType *x, const ValueType *y, ValueType *out) {

  IndexType ind = blockIdx.x * blockDim.x + threadIdx.x;

  if (ind < n)
    out[ind] = y[ind] * x[ind];

}

template <typename ValueType, typename IndexType>
__global__ void kernel_copy_offset_from(const IndexType n, const IndexType src_offset, const IndexType dst_offset,
                                        const ValueType *in, ValueType *out) {

  IndexType ind = blockIdx.x * blockDim.x + threadIdx.x;

  if (ind < n)
    out[ind+dst_offset] = in[ind+src_offset];

}

template <typename ValueType, typename IndexType>
__global__ void kernel_permute(const IndexType n, const IndexType *permute,
                               const ValueType *in, ValueType *out) {

  IndexType ind = blockIdx.x * blockDim.x + threadIdx.x;

  if (ind < n)
    out[permute[ind]] = in[ind];

}

template <typename ValueType, typename IndexType>
__global__ void kernel_permute_backward(const IndexType n, const IndexType *permute,
                                        const ValueType *in, ValueType *out) {

  IndexType ind = blockIdx.x * blockDim.x + threadIdx.x;

  if (ind < n)
    out[ind] = in[permute[ind]];

}

// Reduction
template <unsigned int WARP_SIZE, unsigned int BLOCK_SIZE, typename ValueType, typename IndexType>
__global__ void kernel_reduce(const IndexType n, const ValueType *data, ValueType *out) {

  IndexType gid = blockIdx.x * blockDim.x + threadIdx.x;
  IndexType tid = threadIdx.x;
  IndexType offset = WARP_SIZE * BLOCK_SIZE;

  __shared__ ValueType sdata[BLOCK_SIZE];

  ValueType sum;
  make_ValueType(sum, 0.0);

  for (IndexType i=gid; i<n; i+=offset) {
    sum = sum + data[i];
  }

  sdata[tid] = sum;

  __syncthreads();

#pragma unroll
  for (IndexType i=blockDim.x/2; i>0; i/=2) {

    if (tid < i) {
      sdata[tid] = sdata[tid] + sdata[tid+i];
    }

    __syncthreads();

  }

  if (tid == 0) {
    out[blockIdx.x] = sdata[tid];
  }

}

template <typename ValueType, typename IndexType, unsigned int BLOCK_SIZE>
__global__ void kernel_max(const IndexType n, const ValueType *data, ValueType *out,
                           const IndexType GROUP_SIZE, const IndexType LOCAL_SIZE) {

    IndexType tid = threadIdx.x;

    __shared__ ValueType sdata[BLOCK_SIZE];
    sdata[tid] = ValueType(0);

    // get global id
    IndexType gid = GROUP_SIZE * blockIdx.x + tid;

    for (IndexType i = 0; i < LOCAL_SIZE; ++i, gid += BLOCK_SIZE) {

      if (gid < n) {
        ValueType tmp = data[gid];
        if (tmp > sdata[tid])
          sdata[tid] = tmp;
      }

    }

    __syncthreads();

#pragma unroll
    for (IndexType i = BLOCK_SIZE/2; i > 0; i /= 2) {

      if (tid < i)
        if (sdata[tid+i] > sdata[tid])
          sdata[tid] = sdata[tid+i];

      __syncthreads();

    }

    if (tid == 0)
      out[blockIdx.x] = sdata[tid];

}

template <typename ValueType, typename IndexType>
__global__ void kernel_get_index_values(const IndexType size, const IndexType *index,
                                        const ValueType *in, ValueType *out) {

  IndexType i = blockIdx.x*blockDim.x+threadIdx.x;

  if (i < size)
    out[i] = in[index[i]];

}

template <typename ValueType, typename IndexType>
__global__ void kernel_set_index_values(const IndexType size, const IndexType *index,
                                        const ValueType *in, ValueType *out) {

  IndexType i = blockIdx.x*blockDim.x+threadIdx.x;

  if (i < size)
    out[index[i]] = in[i];

}

template <typename ValueType, typename IndexType, unsigned int BLOCK_SIZE>
__global__ void kernel_amax(const IndexType n, const ValueType *data, ValueType *out,
                            const IndexType GROUP_SIZE, const IndexType LOCAL_SIZE) {

    IndexType tid = threadIdx.x;

    __shared__ ValueType sdata[BLOCK_SIZE];
    sdata[tid] = ValueType(0);

    // get global id
    IndexType gid = GROUP_SIZE * blockIdx.x + tid;

    for (IndexType i = 0; i < LOCAL_SIZE; ++i, gid += BLOCK_SIZE) {

      if (gid < n) {
        ValueType tmp = data[gid];
        tmp = max(tmp, ValueType(-1.0)*tmp);
        if (tmp > sdata[tid])
          sdata[tid] = tmp;
      }

    }

    __syncthreads();

#pragma unroll
    for (IndexType i = BLOCK_SIZE/2; i > 0; i /= 2) {

      if (tid < i) {
        ValueType tmp = sdata[tid+i];
        tmp = max(tmp, ValueType(-1.0)*tmp);
        if (tmp > sdata[tid])
          sdata[tid] = tmp;
      }

      __syncthreads();

    }

    if (tid == 0)
      out[blockIdx.x] = sdata[tid];

}

template <typename IndexType>
__global__ void kernel_powerd(const IndexType n, const double power, double *out) {

  IndexType ind = blockIdx.x * blockDim.x + threadIdx.x;

  if (ind < n)
    out[ind] = pow(out[ind], power);

}

template <typename IndexType>
__global__ void kernel_powerf(const IndexType n, const double power, float *out) {

  IndexType ind = blockIdx.x * blockDim.x + threadIdx.x;

  if (ind < n)
    out[ind] = powf(out[ind], power);

}

template <typename ValueType, typename IndexType>
__global__ void kernel_copy_from_float(const IndexType n, const float *in, ValueType *out) {

  IndexType ind = blockIdx.x * blockDim.x + threadIdx.x;

  if (ind < n)
    out[ind] = ValueType(in[ind]);

}

template <typename ValueType, typename IndexType>
__global__ void kernel_copy_from_double(const IndexType n, const double *in, ValueType *out) {

  IndexType ind = blockIdx.x * blockDim.x + threadIdx.x;

  if (ind < n)
    out[ind] = ValueType(in[ind]);

}


}

#endif
