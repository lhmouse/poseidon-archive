// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_UTILITIES_HPP_
#define POSEIDON_UTILITIES_HPP_

#include "fwd.hpp"
#include "static/async_logger.hpp"
#include "core/abstract_timer.hpp"
#include "static/timer_driver.hpp"
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
do_xlog_format(Log_Level level, const char* file, long line, const char* func,
               const char* templ, const ParamsT&... params)
noexcept
  try {
    // Compose the message.
    ::rocket::tinyfmt_str fmt;
    format(fmt, templ, params...);  // ADL intended
    auto text = fmt.extract_string();

    // Push a new log entry.
    Async_Logger::enqueue(level, file, line, func, ::std::move(text));
    return true;
  }
  catch(exception& stdex) {
    // Ignore this exception, but print a message.
    ::std::fprintf(stderr,
        "WARNING: %s: could not format log: %s\n[exception `%s` thrown from '%s:%ld'\n",
        func, stdex.what(), typeid(stdex).name(), file, line);
    return false;
  }

// Note the format string must be a string literal.
#define POSEIDON_XLOG_(level, ...)  \
    (::poseidon::Async_Logger::is_enabled(level) &&  \
         ::poseidon::do_xlog_format(level, __FILE__, __LINE__, __func__,  \
                                    "" __VA_ARGS__))

#define POSEIDON_LOG_FATAL(...)   POSEIDON_XLOG_(::poseidon::log_level_fatal,  __VA_ARGS__)
#define POSEIDON_LOG_ERROR(...)   POSEIDON_XLOG_(::poseidon::log_level_error,  __VA_ARGS__)
#define POSEIDON_LOG_WARN(...)    POSEIDON_XLOG_(::poseidon::log_level_warn,   __VA_ARGS__)
#define POSEIDON_LOG_INFO(...)    POSEIDON_XLOG_(::poseidon::log_level_info,   __VA_ARGS__)
#define POSEIDON_LOG_DEBUG(...)   POSEIDON_XLOG_(::poseidon::log_level_debug,  __VA_ARGS__)
#define POSEIDON_LOG_TRACE(...)   POSEIDON_XLOG_(::poseidon::log_level_trace,  __VA_ARGS__)

template<typename... ParamsT>
[[noreturn]] ROCKET_NOINLINE
bool
do_xthrow_format(const char* file, long line, const char* func,
                 const char* templ, const ParamsT&... params)
  {
    // Compose the message.
    ::rocket::tinyfmt_str fmt;
    format(fmt, templ, params...);  // ADL intended
    auto text = fmt.extract_string();

    // Push a new log entry.
    if(Async_Logger::is_enabled(log_level_debug))
      Async_Logger::enqueue(log_level_debug, file, line, func, text);

    // Throw the exception.
    ::rocket::sprintf_and_throw<::std::runtime_error>(
        "%s: %s\n[thrown from '%s:%ld']",
        func, text.c_str(), file, line);
  }

#define POSEIDON_THROW(...)  \
    (::poseidon::do_xthrow_format(__FILE__, __LINE__, __func__,  \
                                  "" __VA_ARGS__))

// Creates an asynchronous timer. The timer function will be called by
// the timer thread, so thread safety must be taken into account.
template<typename FuncT>
rcptr<Abstract_Timer>
create_async_timer(int64_t next, int64_t period, FuncT&& func)
  {
    // This is the concrete timer class.
    struct Concrete_Timer : Abstract_Timer
      {
        typename ::std::decay<FuncT>::type m_func;

        explicit
        Concrete_Timer(int64_t next, int64_t period, FuncT&& func)
          : Abstract_Timer(next, period),
            m_func(::std::forward<FuncT>(func))
          { }

        void
        do_on_async_timer(int64_t now)
        override
          { this->m_func(now);  }
      };

    // Allocate an abstract timer and insert it.
    auto timer = ::rocket::make_unique<Concrete_Timer>(next, period,
                                           ::std::forward<FuncT>(func));
    return Timer_Driver::insert(::std::move(timer));
  }

// Creates a one-shot timer. The timer is deleted after being triggered.
template<typename FuncT>
rcptr<Abstract_Timer>
create_async_timer_oneshot(int64_t next, FuncT&& func)
  {
    return noadl::create_async_timer(next, 0, ::std::forward<FuncT>(func));
  }

// Creates a periodic timer.
template<typename FuncT>
rcptr<Abstract_Timer>
create_async_timer_periodic(int64_t period, FuncT&& func)
  {
    return noadl::create_async_timer(period, period, ::std::forward<FuncT>(func));
  }

}  // namespace asteria

#endif
