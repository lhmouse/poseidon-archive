#ifndef POSEIDON_CXX_UTIL_HPP_
#define POSEIDON_CXX_UTIL_HPP_

#define TOKEN_TO_STR_(x_)			# x_
#define TOKEN_TO_STR(x_)			TOKEN_TO_STR_(x_)

#define STRIP_FIRST_(_, ...)		__VA_ARGS__
#define STRIP_FIRST(...)			STRIP_FIRST_(__VA_ARGS__)

#define TOKEN_CAT2_(x_, y_)			x_ ## y_
#define TOKEN_CAT2(x_, y_)			TOKEN_CAT2_(x_, y_)

#define TOKEN_CAT3_(x_, y_, z_)		x_ ## y_ ## z_
#define TOKEN_CAT3(x_, y_, z_)		TOKEN_CAT3_(x_, y_, z_)

#define MAGIC_LN_2_(ln_)			Poseidon_magic_ ## ln_ ## X_
#define MAGIC_LN_1_(ln_)			MAGIC_LN_2_(ln_)

#define UNIQUE_ID					MAGIC_LN_1_(__LINE__)

#include <boost/range/begin.hpp>
#include <boost/range/end.hpp>
#include <boost/range/size.hpp>

#define BEGIN(ar_)					(::boost::begin(ar_))
#define END(ar_)					(::boost::end(ar_))
#define COUNT_OF(ar_)				(::boost::size(ar_))

namespace Poseidon {

template<typename T>
struct Identity {
	typedef T type;
};

}

#endif
