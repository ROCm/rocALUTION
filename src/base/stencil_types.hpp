#ifndef PARALUTION_STENCIL_TYPES_HPP_
#define PARALUTION_STENCIL_TYPES_HPP_

#include <string>

namespace paralution {

/// Stencil Names
const std::string _stencil_type_names [1] = {"Laplace2D"};

/// Stencil Enumeration
enum _stencil_type {Laplace2D = 0};
  
}

#endif // PARALUTION_STENCIL_TYPES_HPP_
