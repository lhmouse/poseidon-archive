// This file is part of Poseidon.
// Copyleft 2022, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_UTILS_
#define POSEIDON_UTILS_

#include "fwd.hpp"
#include <asteria/utils.hpp>
#include <cstdio>

namespace poseidon {

#define POSEIDON_THROW(...)    ASTERIA_THROW(__VA_ARGS__)

using ::rocket::clamp_cast;
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
    return ::rocket::ascii_ci_equal(text.c_str(), text.length(), oth.c_str(), oth.length());
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
explode(cow_vstrings& segments, const cow_string& text, char delim = ',', size_t limit = SIZE_MAX);

size_t
implode(cow_string& text, const cow_vstrings& segments, char delim = ',');

}  // namespace poseidon

#endif
