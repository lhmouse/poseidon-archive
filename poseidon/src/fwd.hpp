// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_FWD_HPP_
#define POSEIDON_FWD_HPP_

#ifdef HAVE_CONFIG_H
#  include <config.h>
#endif

#include <asteria/fwd.hpp>

namespace poseidon {

// Macros
#define POSEIDON_STATIC_CLASS_DECLARE(C)  \
  private:  \
     struct C##_self;  \
     static C##_self* const self;  \
     constexpr C() noexcept = default;  \
     C(const C&) = delete;  \
     C& operator=(const C&) = delete;  \
     ~C() = default  // no semicolon

#define POSEIDON_STATIC_CLASS_DEFINE(C)  \
  template<typename TmIkbXn1>  \
    ROCKET_ARTIFICIAL_FUNCTION static inline  \
    TmIkbXn1* C##_instance()  \
      { static TmIkbXn1 instance[1];  \
        return instance;  }  \
    class C;  \
    struct C::C##_self* const C::self =  \
        C##_instance<struct C::C##_self>();  \
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
using ::asteria::nullopt;

using ::asteria::uptr;
using ::asteria::rcptr;
using ::asteria::cow_vector;
using ::asteria::cow_bivector;
using ::asteria::sso_vector;
using ::asteria::array;
using ::asteria::opt;
using ::asteria::refp;

// Core
class Config_File;

// Singletons
class Main_Config;

}  // namespace poseidon

#endif
