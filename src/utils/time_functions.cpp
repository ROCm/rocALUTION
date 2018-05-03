#include "def.hpp"
#include <stdlib.h>

#include "time_functions.hpp"
#include "../base/backend_manager.hpp"

// the default OS is Linux

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32) || defined(__WIN64) && !defined(__CYGWIN__)
// Windows
#include <windows.h>

#else
// Linux
#include <sys/time.h>

#endif

namespace rocalution {

double rocalution_time(void) {

  double the_time_now = 0.0;

  _rocalution_sync();
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32) || defined(__WIN64) && !defined(__CYGWIN__)
  // Windows

  LARGE_INTEGER now;
  LARGE_INTEGER freq;

  QueryPerformanceCounter(&now);
  QueryPerformanceFrequency(&freq);

  the_time_now = (now.QuadPart*1000000.0) / static_cast<float>(freq.QuadPart);

#else
// Linux

  struct timeval now;

  gettimeofday(&now, NULL);
  the_time_now = now.tv_sec*1000000.0+(now.tv_usec);

#endif

  return the_time_now;

}


}
