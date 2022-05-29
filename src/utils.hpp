// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_UTILS_
#define POSEIDON_UTILS_

#include "fwd.hpp"
#include "static/async_logger.hpp"
#include "core/abstract_timer.hpp"
#include "static/timer_driver.hpp"
#include "core/abstract_async_job.hpp"
#include "core/promise.hpp"
#include "core/future.hpp"
#include "static/worker_pool.hpp"
#include <asteria/utils.hpp>
#include <cstdio>
#include "details/utils.ipp"

namespace poseidon {

using ::asteria::format_string;
using ::asteria::weaken_enum;
using ::asteria::generate_random_seed;
using ::asteria::format_errno;

// Converts all ASCII letters in a string into uppercase.
cow_string
ascii_uppercase(cow_string text);

// Converts all ASCII letters in a string into lowercase.
cow_string
ascii_lowercase(cow_string text);

// Removes all leading and trailing blank characters.
cow_string
ascii_trim(cow_string text);

// Checks whether two strings equal.
template<typename StringT, typename OtherT>
constexpr
bool
ascii_ci_equal(const StringT& text, const OtherT& oth)
  {
    return ::rocket::ascii_ci_equal(
               text.c_str(), text.length(), oth.c_str(), oth.length());
  }

// Checks whether this list contains the specified token.
// Tokens are case-insensitive.
ROCKET_PURE
bool
ascii_ci_has_token(const cow_string& text, char delim, const char* tok, size_t len);

template<typename OtherT>
inline
bool
ascii_ci_has_token(const cow_string& text, char delim, const OtherT& oth)
  {
    return noadl::ascii_ci_has_token(text, delim, oth.c_str(), oth.length());
  }

ROCKET_PURE inline
bool
ascii_ci_has_token(const cow_string& text, const char* tok, size_t len)
  {
    return noadl::ascii_ci_has_token(text, ',', tok, len);
  }

template<typename OtherT>
inline
bool
ascii_ci_has_token(const cow_string& text, const OtherT& oth)
  {
    return noadl::ascii_ci_has_token(text, oth.c_str(), oth.length());
  }

// Split a string into a vector of tokens, and vice versa.
using cow_vstrings = ::rocket::cow_vector<cow_string>;

size_t
explode(cow_vstrings& segments, const cow_string& text, char delim = ',',
        size_t limit = SIZE_MAX);

size_t
implode(cow_string& text, const cow_vstrings& segments, char delim = ',');

// Cast a value using saturation arithmetic.
template<typename ResultT, typename ValueT>
constexpr
ResultT
clamp_cast(const ValueT& value,
           const typename ::std::enable_if<true, ValueT>::type& lower,
           const typename ::std::enable_if<true, ValueT>::type& upper)
  {
    return static_cast<ResultT>(::rocket::clamp(value, lower, upper));
  }

// Composes a string and submits it to the logger.
#define POSEIDON_LOG_X_(level, ...)  \
    (::poseidon::Async_Logger::enabled(level) &&  \
         (::poseidon::details_utils::format_log(level,  \
                __FILE__, __LINE__, __func__, "" __VA_ARGS__), true))

#define POSEIDON_LOG_FATAL(...)   POSEIDON_LOG_X_(::poseidon::log_level_fatal,  __VA_ARGS__)
#define POSEIDON_LOG_ERROR(...)   POSEIDON_LOG_X_(::poseidon::log_level_error,  __VA_ARGS__)
#define POSEIDON_LOG_WARN(...)    POSEIDON_LOG_X_(::poseidon::log_level_warn,   __VA_ARGS__)
#define POSEIDON_LOG_INFO(...)    POSEIDON_LOG_X_(::poseidon::log_level_info,   __VA_ARGS__)
#define POSEIDON_LOG_DEBUG(...)   POSEIDON_LOG_X_(::poseidon::log_level_debug,  __VA_ARGS__)
#define POSEIDON_LOG_TRACE(...)   POSEIDON_LOG_X_(::poseidon::log_level_trace,  __VA_ARGS__)

// Composes a string and throws an exception.
#define POSEIDON_THROW(...)  \
    (::poseidon::details_utils::format_throw(__FILE__, __LINE__, __func__, "" __VA_ARGS__))

// Creates a thread that invokes `loopfnT` repeatedly and never exits.
// Exceptions thrown from the thread procedure are ignored.
template<void loopfnT(void*)>
::pthread_t
create_daemon_thread(const char* name, void* param = nullptr)
  {
    ROCKET_ASSERT_MSG(name, "No thread name specified");
    ROCKET_ASSERT_MSG(::std::strlen(name) <= 15, "Thread name too long");

    // Create the thread.
    ::pthread_t thr;
    int err = ::pthread_create(&thr, nullptr, details_utils::daemon_thread_proc<loopfnT>, param);
    if(err != 0)
      POSEIDON_THROW(
          "could not create thread '$2'\n"
          "[`pthread_create()` failed: $1]",
          format_errno(err), name);

    // Set the thread name. Failure to set the name is ignored.
    err = ::pthread_setname_np(thr, name);
    if(err != 0)
      POSEIDON_LOG_ERROR(
          "Could set thread name '$2'\n"
          "[`pthread_setname_np()` failed: $1]",
          format_errno(err), name);

    // Detach the thread.
    // The thread can't actually exit, but let's be nitpicky.
    ::pthread_detach(thr);
    return thr;
  }

// Creates an asynchronous timer. The timer function will be called by
// the timer thread, so thread safety must be taken into account.
template<typename FuncT>
rcptr<Abstract_Timer>
create_async_timer(int64_t next, int64_t period, FuncT&& func)
  {
    return Timer_Driver::insert(
                 ::rocket::make_unique<details_utils::Timer<
                       typename ::std::decay<FuncT>::type>>(
                           next, period, ::std::forward<FuncT>(func)));
  }

// Creates a one-shot timer.
template<typename FuncT>
rcptr<Abstract_Timer>
create_async_timer_oneshot(int64_t next, FuncT&& func)
  {
    return Timer_Driver::insert(
                 ::rocket::make_unique<details_utils::Timer<
                       typename ::std::decay<FuncT>::type>>(
                           next, 0, ::std::forward<FuncT>(func)));
  }

// Creates a periodic timer.
template<typename FuncT>
rcptr<Abstract_Timer>
create_async_timer_periodic(int64_t period, FuncT&& func)
  {
    return Timer_Driver::insert(
                 ::rocket::make_unique<details_utils::Timer<
                       typename ::std::decay<FuncT>::type>>(
                           period, period, ::std::forward<FuncT>(func)));
  }

// Enqueues an asynchronous job and returns a future to its result.
// Functions with the same key will always be delivered to the same worker.
template<typename FuncT>
futp<typename ::std::result_of<typename ::std::decay<FuncT>::type& ()>::type>
enqueue_async_job(uintptr_t key, FuncT&& func)
  {
    return details_utils::promise(
                 ::rocket::make_unique<details_utils::Async<
                       typename ::std::decay<FuncT>::type>>(
                           key, ::std::forward<FuncT>(func)));
  }

// Enqueues an asynchronous job and returns a future to its result.
// The function is delivered to a random worker.
template<typename FuncT>
futp<typename ::std::result_of<typename ::std::decay<FuncT>::type& ()>::type>
enqueue_async_job(FuncT&& func)
  {
    return details_utils::promise(
                 ::rocket::make_unique<details_utils::Async<
                       typename ::std::decay<FuncT>::type>>(
                           details_utils::random_key, ::std::forward<FuncT>(func)));
  }

}  // namespace poseidon

#endif
