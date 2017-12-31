// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_CHECKED_ARITHMETIC_HPP_
#define POSEIDON_CHECKED_ARITHMETIC_HPP_

#include "cxx_ver.hpp"
#include "exception.hpp"
#include <boost/type_traits/is_unsigned.hpp>
#include <boost/type_traits/is_same.hpp>
#include <boost/static_assert.hpp>

namespace Poseidon {

template<typename T>
inline T checked_add(T lhs, T rhs){
	BOOST_STATIC_ASSERT((boost::is_unsigned<T>::value && !boost::is_same<T, bool>::value));

	const T ret = lhs + rhs;
	if(ret < lhs){
		DEBUG_THROW(Exception, sslit("Integral addition overflow"));
	}
	return ret;
}
template<typename T>
inline T saturated_add(T lhs, T rhs) NOEXCEPT {
	BOOST_STATIC_ASSERT((boost::is_unsigned<T>::value && !boost::is_same<T, bool>::value));

	const T ret = lhs + rhs;
	if(ret < lhs){
		return static_cast<T>(-1);
	}
	return ret;
}

template<typename T>
inline T checked_sub(T lhs, T rhs){
	BOOST_STATIC_ASSERT((boost::is_unsigned<T>::value && !boost::is_same<T, bool>::value));

	const T ret = lhs - rhs;
	if(ret > lhs){
		DEBUG_THROW(Exception, sslit("Integral subtraction overflow"));
	}
	return ret;
}
template<typename T>
inline T saturated_sub(T lhs, T rhs) NOEXCEPT {
	BOOST_STATIC_ASSERT((boost::is_unsigned<T>::value && !boost::is_same<T, bool>::value));

	const T ret = lhs - rhs;
	if(ret > lhs){
		return 0;
	}
	return ret;
}

template<typename T>
inline T checked_mul(T lhs, T rhs){
	BOOST_STATIC_ASSERT((boost::is_unsigned<T>::value && !boost::is_same<T, bool>::value));

	if((lhs == 0) || (rhs == 0)){
		return 0;
	}
	const T ret = lhs * rhs;
	if(ret / lhs != rhs){
		DEBUG_THROW(Exception, sslit("Integral multiplication overflow"));
	}
	return ret;
}
template<typename T>
inline T saturated_mul(T lhs, T rhs) NOEXCEPT {
	BOOST_STATIC_ASSERT((boost::is_unsigned<T>::value && !boost::is_same<T, bool>::value));

	if((lhs == 0) || (rhs == 0)){
		return 0;
	}
	const T ret = lhs * rhs;
	if(ret / lhs != rhs){
		return static_cast<T>(-1);
	}
	return ret;
}

}

#endif
