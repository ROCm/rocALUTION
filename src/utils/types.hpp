#ifndef PARALUTION_UTILS_TYPES_HPP_
#define PARALUTION_UTILS_TYPES_HPP_

#include <assert.h>
#include <limits>

//#define IndexType2 int
#define IndexType2 long
//#define IndexType2 long long

inline static int IndexTypeToInt(const IndexType2 idx) {

  assert (idx <= std::numeric_limits<int>::max());

  return static_cast<int> (idx);

}

#endif // PARALUTION_UTILS_TYPES_HPP_
