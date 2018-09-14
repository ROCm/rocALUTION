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

template <typename F, typename... Ts>
void each_args(F f, Ts&... xs)
{
    (void)std::initializer_list<int>{((void)f(xs), 0)...};
}

struct log_arg
{
    log_arg(std::ostream& os, std::string& separator) : os_(os), separator_(separator) {}

    /// Generic overload for () operator.
    template <typename T>
    void operator()(T& x) const
    {
        os_ << separator_ << x;
    }

    private:
    std::ostream& os_;
    std::string& separator_;
};

template <typename P, typename F, typename... Ts>
void log_arguments(std::ostream& os, std::string& separator, int rank, P ptr, F fct, Ts&... xs)
{
    os << "\n[rank:" << rank << "]# ";
    os << "Obj addr: " << ptr << "; ";
    os << "fct: " << fct;
    each_args(log_arg{os, separator}, xs...);
}

template <typename P, typename F, typename... Ts>
void log_debug(P ptr, F fct, Ts&... xs)
{
    if(_get_backend_descriptor()->log_file != NULL)
    {
        std::string comma_separator = ", ";
        std::ostream* os = _get_backend_descriptor()->log_file;
        log_arguments(*os,
                      comma_separator,
                      _get_backend_descriptor()->rank,
                      ptr,
                      fct,
                      xs...);
    }
}

} // namespace rocalution

// Do not edit
#ifdef DEBUG_MODE

#undef VERBOSE_LEVEL
#define VERBOSE_LEVEL 10

#endif

// Do not edit
#define LOG_STREAM std::cout

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
