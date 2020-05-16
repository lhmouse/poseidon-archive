// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_UTILITIES_HPP_
#define POSEIDON_UTILITIES_HPP_

#include "fwd.hpp"
#include "static/async_logger.hpp"
#include <asteria/utilities.hpp>
#include <cstdio>

namespace poseidon {

using ::asteria::utf8_encode;
using ::asteria::utf8_decode;
using ::asteria::utf16_encode;
using ::asteria::utf16_decode;

using ::asteria::format_string;
using ::asteria::weaken_enum;
using ::asteria::generate_random_seed;
using ::asteria::format_errno;

template<typename... ParamsT>
ROCKET_NOINLINE
bool
do_xlog_format(Async_Logger::Level level, const char* file, long line,
               const char* func, const ParamsT&... params)
noexcept
  try {
    // Compose the message.
    ::rocket::tinyfmt_str fmt;
    format(fmt, params...);  // ADL intended
    auto text = fmt.extract_string();

    // Push a new entry.
    Async_Logger::write(level, file, line, func, ::std::move(text));
    return true;
  }
  catch(exception& stdex) {
    // Ignore this exception, but print a message.
    ::std::fprintf(stderr,
        "WARNING: %s: could not format log: %s\n"
        "[exception `%s` thrown from '%s:%ld'\n",
        func, stdex.what(),
        typeid(stdex).name(), file, line);
    return false;
  }

// Note the format string must be a string literal.
#define POSEIDON_XLOG_(level, ...)  \
            (::poseidon::Async_Logger::is_enabled(level) &&  \
             ::poseidon::do_xlog_format(level, __FILE__, __LINE__, __func__, "" __VA_ARGS__))

#define POSEIDON_LOG_FATAL(...)   POSEIDON_XLOG_(::poseidon::Async_Logger::level_fatal,  __VA_ARGS__)
#define POSEIDON_LOG_ERROR(...)   POSEIDON_XLOG_(::poseidon::Async_Logger::level_error,  __VA_ARGS__)
#define POSEIDON_LOG_WARN(...)    POSEIDON_XLOG_(::poseidon::Async_Logger::level_warn,   __VA_ARGS__)
#define POSEIDON_LOG_INFO(...)    POSEIDON_XLOG_(::poseidon::Async_Logger::level_info,   __VA_ARGS__)
#define POSEIDON_LOG_DEBUG(...)   POSEIDON_XLOG_(::poseidon::Async_Logger::level_debug,  __VA_ARGS__)
#define POSEIDON_LOG_TRACE(...)   POSEIDON_XLOG_(::poseidon::Async_Logger::level_trace,  __VA_ARGS__)

}  // namespace asteria

#endif
