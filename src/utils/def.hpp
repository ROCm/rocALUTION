#ifndef ROCALUTION_UTILS_DEF_HPP_
#define ROCALUTION_UTILS_DEF_HPP_

// Uncomment to define verbose level
#define VERBOSE_LEVEL 2

// Uncomment for debug mode
// #define DEBUG_MODE

// Uncomment to disable the assert()s
// #define ASSERT_OFF

// Uncomment to log only on specific MPI rank
// When logging into a file, this will be unset
#define LOG_MPI_RANK 0

// Comment to enable automatic object tracking
#define OBJ_TRACKING_OFF

// ******************
// ******************
// Do not edit below!
// ******************
// ******************

#ifdef ASSERT_OFF
#define assert(a) ;
#else
#ifdef NDEBUG
#undef NDEBUG
#include <assert.h>
#define NDEBUG
#else
#include <assert.h>
#endif
#endif

#ifdef DEBUG_MODE
#define assert_dbg(a) assert(a)
#else
#define assert_dbg(a) ;
#endif

// TODO #define SUPPORT_COMPLEX

#endif // ROCALUTION_UTILS_DEF_HPP_
