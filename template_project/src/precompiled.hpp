#ifndef TEMPLATE_PROJECT_PRECOMPILED_HPP_
#define TEMPLATE_PROJECT_PRECOMPILED_HPP_

#include <poseidon/precompiled.hpp>

#include <poseidon/shared_nts.hpp>
#include <poseidon/exception.hpp>
#include <poseidon/log.hpp>
#include <poseidon/profiler.hpp>
#include <poseidon/errno.hpp>
#include <poseidon/time.hpp>
#include <poseidon/random.hpp>
#include <poseidon/flags.hpp>
#include <poseidon/module_raii.hpp>
#include <poseidon/uuid.hpp>
#include <poseidon/endian.hpp>
#include <poseidon/string.hpp>
#include <poseidon/checked_arithmetic.hpp>

#include "log.hpp"

#include <cstdint>
#include <array>
#include <type_traits>
#include <typeinfo>

#include <boost/container/flat_map.hpp>
#include <boost/container/flat_set.hpp>

namespace TemplateProject {

using Poseidon::Exception;
using Poseidon::SharedNts;

using Poseidon::sslit;

using Poseidon::checked_add;
using Poseidon::saturated_add;
using Poseidon::checked_sub;
using Poseidon::saturated_sub;
using Poseidon::checked_mul;
using Poseidon::saturated_mul;

}

#endif
