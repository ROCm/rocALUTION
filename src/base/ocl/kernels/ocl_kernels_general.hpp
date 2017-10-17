#ifndef PARALUTION_OCL_KERNELS_GENERAL_HPP_
#define PARALUTION_OCL_KERNELS_GENERAL_HPP_

namespace paralution {

const char *ocl_kernels_general = CL_KERNEL(

__kernel void kernel_red_recurse(__global       IndexType *dst,
                                 __global const IndexType *src,
                                          const IndexType numElems) {

  IndexType index = BLOCK_SIZE * get_global_id(0);

  if (index >= numElems)
    return;

  IndexType i = index;

  if (i < BLOCK_SIZE)
    return;

  IndexType a = 0;

  while (i >= BLOCK_SIZE) {
    a += src[i];
    i -= BLOCK_SIZE;
  }

  dst[index] = a;

}

__kernel void kernel_red_partial_sum(__global       IndexType *dst,
                                     __global const IndexType *src,
                                              const IndexType numElems,
                                              const IndexType shift) {

  IndexType index = get_global_id(0);
  IndexType tid   = get_local_id(0);
  IndexType gid   = get_group_id(0);

  if (index < numElems) {

    __local IndexType data[BLOCK_SIZE];

    data[tid] = src[index];

    barrier(CLK_LOCAL_MEM_FENCE);

    for (IndexType i = BLOCK_SIZE/2; i > 0; i/=2) {

      if (tid < i)
        data[tid] = data[tid] + data[tid+i];

      barrier(CLK_LOCAL_MEM_FENCE);

    }

    if (tid == 0 && BLOCK_SIZE*(1+gid)-1<numElems)
      dst[BLOCK_SIZE*(1+gid)-1+shift] = data[0];

  }

}

__kernel void kernel_red_extrapolate(__global       IndexType *dst,
                                     __global const IndexType *srcBorder,
                                     __global const IndexType *srcData,
                                                    IndexType  numElems,
                                              const IndexType  shift) {

  IndexType index = get_local_size(0) * get_local_id(0);

  if (index < numElems-1) {

    IndexType sum = srcBorder[index];

    for(IndexType i = 0; i < get_local_size(0) && index+i<numElems; ++i) {
      sum += srcData[index+i];
      dst[index+i+shift] = sum;
    }

  }

}

);

}

#endif // PARALUTION_OCL_KERNELS_GENERAL_HPP_
