#ifndef ROCALUTION_UTILS_LOG_HPP_
#define ROCALUTION_UTILS_LOG_HPP_

#include "def.hpp"
#include "../base/backend_manager.hpp"

#include <iostream>
#include <stdlib.h>
#include <string>
#include <sstream>

namespace rocalution {

void _rocalution_open_log_file(void);
void _rocalution_close_log_file(void);

} // namespace rocalution

// Do not edit
#ifdef DEBUG_MODE

#undef VERBOSE_LEVEL
#define VERBOSE_LEVEL 10

#endif

// Do not edit
#ifdef LOG_FILE

#define LOG_STREAM                                             \
    ((_get_backend_descriptor()->log_file == NULL) ? std::cout \
                                                   : *(_get_backend_descriptor()->log_file))

#else

#define LOG_STREAM std::cout

#endif

// LOG ERROR
#define FATAL_ERROR(file, line)                                    \
    {                                                              \
        LOG_INFO("Fatal error - the program will be terminated "); \
        LOG_INFO("File: " << file << "; line: " << line);          \
        exit(1);                                                   \
    }

// LOG VERBOSE
#ifdef VERBOSE_LEVEL

#define LOG_VERBOSE_INFO(level, stream) \
    {                                   \
        if(level <= VERBOSE_LEVEL)      \
            LOG_INFO(stream);           \
    }

#else

#define LOG_VERBOSE_INFO(level, stream) ;

#endif

// LOG DEBUG
#ifdef DEBUG_MODE

#define LOG_DEBUG(obj, fct, stream)                                           \
    {                                                                         \
        LOG_INFO("# Obj addr: " << obj << "; fct: " << fct << " " << stream); \
    }

#else

#define LOG_DEBUG(obj, fct, stream) ;

#endif

#ifdef LOG_MPI_RANK

#define LOG_INFO(stream)                                    \
    {                                                       \
        if(_get_backend_descriptor()->rank == LOG_MPI_RANK) \
            LOG_STREAM << stream << std::endl;              \
    }

#else // LOG_MPI_RANK

#define LOG_INFO(stream)                                                                         \
    {                                                                                            \
        LOG_STREAM << "[rank:" << _get_backend_descriptor()->rank << "]" << stream << std::endl; \
    }

#endif // LOG_MPI_RANK

#endif // ROCALUTION_UTILS_LOG_HPP_
