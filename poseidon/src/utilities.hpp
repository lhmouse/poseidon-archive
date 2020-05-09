// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_UTILITIES_HPP_
#define POSEIDON_UTILITIES_HPP_

#include "fwd.hpp"
#include <asteria/utilities.hpp>

namespace poseidon {

using ::asteria::utf8_encode;
using ::asteria::utf8_decode;
using ::asteria::utf16_encode;
using ::asteria::utf16_decode;

using ::asteria::format_string;
using ::asteria::weaken_enum;
using ::asteria::generate_random_seed;
using ::asteria::format_errno;

}  // namespace asteria

#endif
