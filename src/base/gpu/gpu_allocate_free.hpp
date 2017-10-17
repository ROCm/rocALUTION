#ifndef PARALUTION_GPU_ALLOCATE_FREE_HPP_
#define PARALUTION_GPU_ALLOCATE_FREE_HPP_

#include <iostream>

namespace paralution {

template <typename DataType>
void allocate_gpu(const int size, DataType **ptr);

template <typename DataType>
void free_gpu(DataType **ptr);

template <typename DataType>
void set_to_zero_gpu(const int blocksize,
                     const int max_threads,
                     const int size, DataType *ptr);

template <typename DataType>
void set_to_one_gpu(const int blocksize,
                    const int max_threads,
                    const int size, DataType *ptr);


}

#endif // PARALUTION_GPU_ALLOCATE_FREE_HPP_
