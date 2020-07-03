// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_FWD_HPP_
#define POSEIDON_FWD_HPP_

#ifdef HAVE_CONFIG_H
#  include <config.h>
#endif

#include <asteria/fwd.hpp>
#include <rocket/linear_buffer.hpp>
#include <rocket/atomic.hpp>
#include <rocket/mutex.hpp>
#include <rocket/recursive_mutex.hpp>
#include <rocket/condition_variable.hpp>
#include <rocket/once_flag.hpp>
#include <string>
#include <vector>
#include <deque>
#include <unistd.h>

namespace poseidon {
namespace noadl = poseidon;

// Macros
#define POSEIDON_STATIC_CLASS_DECLARE(C)  \
    private:  \
      struct __attribute__((__visibility__("hidden"))) C##_self;  \
      static C##_self* const self;  \
      \
      constexpr C() noexcept = default;  \
      C(const C&) = delete;  \
      C& operator=(const C&) = delete;  \
      ~C() = default  // no semicolon

#define POSEIDON_STATIC_CLASS_DEFINE(C)  \
    template<typename TmIkbXn1>  \
    ROCKET_ARTIFICIAL_FUNCTION static inline  \
    TmIkbXn1* C##_inst()  \
      {  \
        static TmIkbXn1 instance[1] = { };  \
        return instance;  \
      }  \
    \
    class C;  \
    struct C::C##_self* const C::self = C##_inst<C::C##_self>();  \
    struct C::C##_self : C  \
      // add members here

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

using ::asteria::nullopt_t;
using ::asteria::cow_string;
using ::asteria::cow_u16string;
using ::asteria::cow_u32string;
using ::asteria::phsh_string;
using ::asteria::tinybuf;
using ::asteria::tinyfmt;

using ::asteria::cbegin;
using ::asteria::cend;
using ::asteria::begin;
using ::asteria::end;
using ::asteria::swap;
using ::asteria::xswap;
using ::asteria::size;

using ::asteria::uptr;
using ::asteria::rcptr;
using ::asteria::rcfwdp;
using ::asteria::array;
using ::asteria::opt;
using ::asteria::refp;

using ::rocket::linear_buffer;
using ::rocket::atomic;
using ::rocket::atomic_relaxed;
using ::rocket::atomic_acq_rel;
using ::rocket::atomic_seq_cst;
using ::rocket::mutex;
using ::rocket::recursive_mutex;
using ::rocket::condition_variable;
using ::rocket::once_flag;

using ::asteria::nullopt;

struct FD_Closer
  {
    constexpr
    int
    null()
    const noexcept
      { return -1;  }

    constexpr
    bool
    is_null(int fd)
    const noexcept
      { return fd == -1;  }

    void
    close(int fd)
      { ::close(fd);  }
  };

using unique_FD = ::rocket::unique_handle<int, FD_Closer>;
static_assert(sizeof(unique_FD) == sizeof(int));

using atomic_signal = rocket::atomic_relaxed<int>;

// Core
class Config_File;
class Abstract_Timer;
class Abstract_Async_Job;
class Abstract_Future;
class Abstract_Fiber;
class URL;

template<typename V> class Promise;
template<typename V> class Future;

template<typename V> using prom = Promise<V>;
template<typename V> using futp = rcptr<const Future<V>>;

// Socket
enum IO_Result : ptrdiff_t;
enum Connection_State : uint8_t;
enum Socket_Address_Class : uint8_t;

struct SSL_deleter;
struct SSL_CTX_deleter;

class Socket_Address;
class Abstract_Socket;
class Abstract_Listen_Socket;
class Abstract_Stream_Socket;
class Abstract_TCP_Socket;
class Abstract_TCP_Server_Socket;
class Abstract_TCP_Client_Socket;
class Abstract_TLS_Socket;
class Abstract_TLS_Server_Socket;
class Abstract_TLS_Client_Socket;
class Abstract_UDP_Socket;

// Singletons
class Main_Config;
class Async_Logger;
class Timer_Driver;
class Network_Driver;
class Worker_Pool;
class Fiber_Scheduler;

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
