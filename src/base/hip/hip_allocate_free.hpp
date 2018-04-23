#ifndef PARALUTION_HIP_ALLOCATE_FREE_HPP_
#define PARALUTION_HIP_ALLOCATE_FREE_HPP_

#include <iostream>

namespace paralution {

template <typename DataType>
void allocate_hip(const int size, DataType **ptr);

template <typename DataType>
void free_hip(DataType **ptr);

template <typename DataType>
void set_to_zero_hip(const int blocksize,
                     const int max_threads,
                     const int size, DataType *ptr);

template <typename DataType>
void set_to_one_hip(const int blocksize,
                    const int max_threads,
                    const int size, DataType *ptr);


}

#endif // PARALUTION_HIP_ALLOCATE_FREE_HPP_
