// This file is part of Poseidon.
// Copyleft 2022, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_PRECOMPILED_
#define POSEIDON_PRECOMPILED_

#include "version.h"

#ifdef HAVE_CONFIG_H
#  include <config.h>
#endif

#include <rocket/cow_string.hpp>
#include <rocket/cow_vector.hpp>
#include <rocket/cow_hashmap.hpp>
#include <rocket/static_vector.hpp>
#include <rocket/prehashed_string.hpp>
#include <rocket/unique_handle.hpp>
#include <rocket/unique_posix_file.hpp>
#include <rocket/unique_posix_dir.hpp>
#include <rocket/unique_posix_fd.hpp>
#include <rocket/variant.hpp>
#include <rocket/optional.hpp>
#include <rocket/array.hpp>
#include <rocket/reference_wrapper.hpp>
#include <rocket/tinyfmt.hpp>
#include <rocket/tinyfmt_str.hpp>
#include <rocket/tinyfmt_file.hpp>
#include <rocket/ascii_numget.hpp>
#include <rocket/ascii_numput.hpp>
#include <rocket/format.hpp>
#include <rocket/atomic.hpp>
#include <rocket/ascii_case.hpp>
#include <rocket/mutex.hpp>
#include <rocket/recursive_mutex.hpp>
#include <rocket/condition_variable.hpp>
#include <rocket/once_flag.hpp>

#include <iterator>
#include <memory>
#include <utility>
#include <exception>
#include <typeinfo>
#include <type_traits>
#include <functional>
#include <algorithm>
#include <array>
#include <string>
#include <vector>
#include <deque>
#include <bitset>

#include <cstdio>
#include <climits>
#include <cmath>
#include <cfenv>
#include <cfloat>
#include <cstring>
#include <cerrno>

#include <sys/types.h>
#include <unistd.h>
#include <endian.h>

#endif
