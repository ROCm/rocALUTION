#ifndef PARALUTION_UTILS_DEF_HPP_
#define PARALUTION_UTILS_DEF_HPP_

// Uncomment to define verbose level
#define VERBOSE_LEVEL 2


// Uncomment for debug mode
// #define DEBUG_MODE

// Uncomment to log all msg to file
// #define LOG_FILE

// Uncomment to disable the assert()s
// #define ASSERT_OFF

// Uncomment to enable encryption for the log msg
// #define LOG_ENC

// Uncomment to log only on specific MPI rank
// e.g. LOG_MPI_RANK 0 will log on rank=0 only;
// When logging into a file, this will be unset
#define LOG_MPI_RANK 0

// Uncomment to disable automatic object tracking
// #define OBJ_TRACKING_OFF




// ******************
// ******************
// Do not edit below! 
// ******************
// ******************

#ifdef ASSERT_OFF

#define assert(a) ;

#else

#include <assert.h>

#endif


#ifdef DEBUG_MODE

#define assert_dbg(a) assert(a)

#else

#define assert_dbg(a) ;

#endif

#define SUPPORT_COMPLEX

#ifdef LOG_FILE

#undef LOG_MPI_RANK

#endif

#endif // PARALUTION_UTILS_DEF_HPP_
