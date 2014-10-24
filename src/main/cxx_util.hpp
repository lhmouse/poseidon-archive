#ifndef POSEIDON_CXX_UTIL_HPP_
#define POSEIDON_CXX_UTIL_HPP_

#include <cstddef>
#include <boost/range/begin.hpp>
#include <boost/range/end.hpp>
#include <boost/range/size.hpp>

#define COUNT_OF(ar)				(::boost::size(ar))
#define BEGIN(ar)					(::boost::begin(ar))
#define END(ar)						(::boost::end(ar))

#define TOKEN_TO_STR_(x_)			# x_
#define TOKEN_TO_STR(x_)			TOKEN_TO_STR_(x_)

#define STRIP_FIRST_(_, ...)		__VA_ARGS__
#define STRIP_FIRST(...)			STRIP_FIRST_(__VA_ARGS__)

#define MAGIC_LN_2_(ln_)			Poseidon_magic_ ## ln_ ## _
#define MAGIC_LN_1_(ln_)			MAGIC_LN_2_(ln_)

#define UNIQUE_ID					MAGIC_LN_1_(__LINE__)

namespace Poseidon {

template<typename T>
struct Identity {
	typedef T type;
};

}

#endif
