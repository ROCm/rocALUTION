#ifndef ROCALUTION_UTILS_ALLOCATE_FREE_HPP_
#define ROCALUTION_UTILS_ALLOCATE_FREE_HPP_

namespace rocalution {

/// Allocate buffer on the host
template <typename DataType>
void allocate_host(int size, DataType** ptr);

/// Free buffer on the host
template <typename DataType>
void free_host(DataType** ptr);

/// set a buffer to zero on the host
template <typename DataType>
void set_to_zero_host(int size, DataType* ptr);

} // namespace rocalution

#endif // ROCALUTION_UTILS_ALLOCATE_FREE_HPP_
