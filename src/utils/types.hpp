/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#ifndef ROCALUTION_UTILS_TYPES_HPP_
#define ROCALUTION_UTILS_TYPES_HPP_

#include "def.hpp"

#include <limits>

//#define IndexType2 int
#define IndexType2 long
//#define IndexType2 long long

namespace rocalution {

inline static int IndexTypeToInt(const IndexType2 idx)
{
    assert(idx <= std::numeric_limits<int>::max());

    return static_cast<int>(idx);
}

} // namespace rocalution

#endif // ROCALUTION_UTILS_TYPES_HPP_
