// This file is part of Poseidon.
// Copyleft 2022, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_FWD_
#define POSEIDON_FWD_

#include "version.h"
#include <asteria/fwd.hpp>
#include <rocket/ascii_case.hpp>
#include <rocket/linear_buffer.hpp>
#include <rocket/atomic.hpp>
#include <rocket/mutex.hpp>
#include <rocket/recursive_mutex.hpp>
#include <rocket/condition_variable.hpp>
#include <rocket/once_flag.hpp>
#include <rocket/tinyfmt_str.hpp>
#include <rocket/tinyfmt_file.hpp>
#include <array>
#include <vector>
#include <deque>
#include <unordered_map>

namespace poseidon {
namespace noadl = poseidon;

// Aliases
using ::std::initializer_list;
using ::std::nullptr_t;
using ::std::max_align_t;
using ::std::int8_t;
using ::std::uint8_t;
using ::std::int16_t;
using ::std::uint16_t;
using ::std::int32_t;
using ::std::uint32_t;
using ::std::int64_t;
using ::std::uint64_t;
using ::std::intptr_t;
using ::std::uintptr_t;
using ::std::intmax_t;
using ::std::uintmax_t;
using ::std::ptrdiff_t;
using ::std::size_t;
using ::std::wint_t;
using ::std::exception;
using ::std::exception_ptr;
using ::std::type_info;
using ::std::pair;
using ::std::unique_ptr;
using ::std::shared_ptr;
using ::std::weak_ptr;
using ::std::array;
using ::std::vector;
using ::std::deque;
using ::std::unordered_map;

using ::std::static_pointer_cast;
using ::std::dynamic_pointer_cast;
using ::std::const_pointer_cast;

using ::rocket::linear_buffer;
using ::rocket::atomic;
using ::rocket::atomic_relaxed;
using ::rocket::atomic_acq_rel;
using ::rocket::atomic_seq_cst;
using atomic_signal = ::rocket::atomic_relaxed<int>;
using plain_mutex = ::rocket::mutex;
using ::rocket::recursive_mutex;
using ::rocket::condition_variable;
using ::rocket::once_flag;
using ::rocket::cow_vector;
using ::rocket::cow_hashmap;
using ::rocket::static_vector;
using ::rocket::cow_string;
using ::rocket::cow_u16string;
using ::rocket::cow_u32string;
using ::rocket::tinybuf;
using ::rocket::tinyfmt;
using ::rocket::tinyfmt_str;
using ::rocket::tinyfmt_file;

using ::rocket::begin;
using ::rocket::end;
using ::rocket::swap;
using ::rocket::xswap;
using ::rocket::size;
using ::rocket::nullopt;
using ::rocket::nullopt_t;
using ::rocket::sref;
using ::rocket::optional;
using ::rocket::variant;
using phsh_string = ::rocket::prehashed_string;

#define POSEIDON_HIDDEN_STRUCT(CLASS, MEMBER)  \
  using CLASS##_##MEMBER = MEMBER;  \
  struct __attribute__((__visibility__("hidden")))  \
      CLASS::MEMBER : CLASS##_##MEMBER { }  // no semicolon

// Core types
enum Async_State : uint8_t;
enum Future_State : uint8_t;
class Config_File;
class Abstract_Timer;
class Abstract_Future;
template<typename ValueT> class Future;
class Abstract_Fiber;
class Abstract_Task;

// Manager types and instances
extern atomic_signal exit_signal;
extern class Main_Config& main_config;
extern class Async_Logger& async_logger;
extern class Timer_Driver& timer_driver;
extern class Fiber_Scheduler& fiber_scheduler;
extern class Task_Executor& task_executor;

// Log levels
// Note each level has a hardcoded name and number.
// Don't change their values or reorder them.
enum Log_Level : uint8_t
  {
    log_level_fatal  = 0,
    log_level_error  = 1,
    log_level_warn   = 2,
    log_level_info   = 3,
    log_level_debug  = 4,
    log_level_trace  = 5,
  };

// Composes a string and submits it to the logger.
// Note that in order to use these macros, you still have to include
// <poseidon/static/async_logger.hpp>; otherwise there may be errors
// about incomplete types.
#define POSEIDON_LOG_GENERIC(LEVEL, TEMPLATE, ...)  \
  (::poseidon::async_logger.enabled(::poseidon::log_level_##LEVEL)  \
   &&  \
   ([=](const char* f5zuNP3w) -> bool  \
       __attribute__((__noinline__, __nothrow__, __cold__))  \
     {  \
       try {  \
         ::poseidon::Async_Logger::Queued_Message iQw3Zbsf;  \
         iQw3Zbsf.level = ::poseidon::log_level_##LEVEL;  \
         iQw3Zbsf.file = __FILE__;  \
         iQw3Zbsf.line = __LINE__;  \
         iQw3Zbsf.func = f5zuNP3w;  \
         try {  \
           iQw3Zbsf.text = ::asteria::format_string(  \
               (::asteria::make_string_template TEMPLATE), ##__VA_ARGS__);  \
         }  \
         catch(::std::exception& xSG022wB) {  \
           iQw3Zbsf.text = ::asteria::format_string(  \
               "Failed to compose log message: $1", xSG022wB);  \
         }  \
         ::poseidon::async_logger.enqueue(::std::move(iQw3Zbsf));  \
         \
         if(iQw3Zbsf.level <= ::poseidon::log_level_error)  \
           ::poseidon::async_logger.synchronize();  \
       }  \
       catch(::std::exception& aJHPhv84) {  \
         ::fprintf(stderr,  \
             "%s: Error writing log:\n  %s\n", f5zuNP3w, aJHPhv84.what());  \
       }  \
       return true;  \
     }  \
     (__func__)))

#define POSEIDON_LOG_FATAL(...)   POSEIDON_LOG_GENERIC(fatal, __VA_ARGS__)
#define POSEIDON_LOG_ERROR(...)   POSEIDON_LOG_GENERIC(error, __VA_ARGS__)
#define POSEIDON_LOG_WARN(...)    POSEIDON_LOG_GENERIC(warn,  __VA_ARGS__)
#define POSEIDON_LOG_INFO(...)    POSEIDON_LOG_GENERIC(info,  __VA_ARGS__)
#define POSEIDON_LOG_DEBUG(...)   POSEIDON_LOG_GENERIC(debug, __VA_ARGS__)
#define POSEIDON_LOG_TRACE(...)   POSEIDON_LOG_GENERIC(trace, __VA_ARGS__)

}  // namespace poseidon

#endif
