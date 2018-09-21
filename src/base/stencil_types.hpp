/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#ifndef ROCALUTION_STENCIL_TYPES_HPP_
#define ROCALUTION_STENCIL_TYPES_HPP_

#include <string>

namespace rocalution {

/// Stencil Names
const std::string _stencil_type_names[1] = {"Laplace2D"};

/// Stencil Enumeration
enum _stencil_type
{
    Laplace2D = 0
};

} // namespace rocalution

#endif // ROCALUTION_STENCIL_TYPES_HPP_
