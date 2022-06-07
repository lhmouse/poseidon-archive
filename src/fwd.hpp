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
#include <string>
#include <vector>
#include <deque>

namespace poseidon {
namespace noadl = poseidon;

// Aliases
using ::std::initializer_list;
using ::std::integer_sequence;
using ::std::index_sequence;
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
using ::std::type_info;
using ::std::pair;
using ::std::unique_ptr;
using ::std::shared_ptr;
using ::std::weak_ptr;
using ::std::array;

using ::std::static_pointer_cast;
using ::std::dynamic_pointer_cast;
using ::std::const_pointer_cast;

using ::rocket::linear_buffer;
using ::rocket::atomic;
using ::rocket::atomic_relaxed;
using ::rocket::atomic_acq_rel;
using ::rocket::atomic_seq_cst;
using atomic_signal = atomic_relaxed<int>;
using simple_mutex = ::rocket::mutex;
using ::rocket::recursive_mutex;
using ::rocket::condition_variable;
using ::rocket::once_flag;

using ::asteria::nullopt_t;
using ::asteria::cow_string;
using ::asteria::cow_u16string;
using ::asteria::cow_u32string;
using ::asteria::phsh_string;
using ::asteria::tinybuf;
using ::asteria::tinyfmt;

using ::asteria::begin;
using ::asteria::end;
using ::asteria::swap;
using ::asteria::xswap;
using ::asteria::size;
using ::asteria::nullopt;
using ::asteria::sref;

// Core
class Config_File;

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

// Future states
enum Future_State : uint8_t
  {
    future_state_empty   = 0,
    future_state_value   = 1,
    future_state_except  = 2,
  };

// Asynchronous function states
enum Async_State : uint8_t
  {
    async_state_initial    = 0,
    async_state_pending    = 1,
    async_state_suspended  = 2,
    async_state_running    = 3,
    async_state_finished   = 4,
  };

}  // namespace poseidon

#endif
