// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2017, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_CHECKED_ARITHMETIC_HPP_
#define POSEIDON_CHECKED_ARITHMETIC_HPP_

#include "cxx_ver.hpp"
#include "exception.hpp"
#include <boost/type_traits/is_unsigned.hpp>
#include <boost/type_traits/is_same.hpp>
#include <boost/static_assert.hpp>

namespace Poseidon {

template<typename T>
T checked_add(T lhs, T rhs){
	BOOST_STATIC_ASSERT((boost::is_unsigned<T>::value && !boost::is_same<T, bool>::value));

	const T ret = lhs + rhs;
	if(ret < lhs){
		DEBUG_THROW(Exception, sslit("Integral addition overflow"));
	}
	return ret;
}

template unsigned char      checked_add<unsigned char     >(unsigned char      lhs, unsigned char      rhs);
template unsigned short     checked_add<unsigned short    >(unsigned short     lhs, unsigned short     rhs);
template unsigned int       checked_add<unsigned int      >(unsigned int       lhs, unsigned int       rhs);
template unsigned long      checked_add<unsigned long     >(unsigned long      lhs, unsigned long      rhs);
template unsigned long long checked_add<unsigned long long>(unsigned long long lhs, unsigned long long rhs);

template<typename T>
T saturated_add(T lhs, T rhs) NOEXCEPT {
	BOOST_STATIC_ASSERT((boost::is_unsigned<T>::value && !boost::is_same<T, bool>::value));

	const T ret = lhs + rhs;
	if(ret < lhs){
		return static_cast<T>(-1);
	}
	return ret;
}

template unsigned char      saturated_add<unsigned char     >(unsigned char      lhs, unsigned char      rhs) NOEXCEPT;
template unsigned short     saturated_add<unsigned short    >(unsigned short     lhs, unsigned short     rhs) NOEXCEPT;
template unsigned int       saturated_add<unsigned int      >(unsigned int       lhs, unsigned int       rhs) NOEXCEPT;
template unsigned long      saturated_add<unsigned long     >(unsigned long      lhs, unsigned long      rhs) NOEXCEPT;
template unsigned long long saturated_add<unsigned long long>(unsigned long long lhs, unsigned long long rhs) NOEXCEPT;

template<typename T>
T checked_sub(T lhs, T rhs){
	BOOST_STATIC_ASSERT((boost::is_unsigned<T>::value && !boost::is_same<T, bool>::value));

	const T ret = lhs - rhs;
	if(ret > lhs){
		DEBUG_THROW(Exception, sslit("Integral subtraction overflow"));
	}
	return ret;
}

template unsigned char      checked_sub<unsigned char     >(unsigned char      lhs, unsigned char      rhs);
template unsigned short     checked_sub<unsigned short    >(unsigned short     lhs, unsigned short     rhs);
template unsigned int       checked_sub<unsigned int      >(unsigned int       lhs, unsigned int       rhs);
template unsigned long      checked_sub<unsigned long     >(unsigned long      lhs, unsigned long      rhs);
template unsigned long long checked_sub<unsigned long long>(unsigned long long lhs, unsigned long long rhs);

template<typename T>
T saturated_sub(T lhs, T rhs) NOEXCEPT {
	BOOST_STATIC_ASSERT((boost::is_unsigned<T>::value && !boost::is_same<T, bool>::value));

	const T ret = lhs - rhs;
	if(ret > lhs){
		return 0;
	}
	return ret;
}

template unsigned char      saturated_sub<unsigned char     >(unsigned char      lhs, unsigned char      rhs) NOEXCEPT;
template unsigned short     saturated_sub<unsigned short    >(unsigned short     lhs, unsigned short     rhs) NOEXCEPT;
template unsigned int       saturated_sub<unsigned int      >(unsigned int       lhs, unsigned int       rhs) NOEXCEPT;
template unsigned long      saturated_sub<unsigned long     >(unsigned long      lhs, unsigned long      rhs) NOEXCEPT;
template unsigned long long saturated_sub<unsigned long long>(unsigned long long lhs, unsigned long long rhs) NOEXCEPT;

template<typename T>
T checked_mul(T lhs, T rhs){
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

template unsigned char      checked_mul<unsigned char     >(unsigned char      lhs, unsigned char      rhs);
template unsigned short     checked_mul<unsigned short    >(unsigned short     lhs, unsigned short     rhs);
template unsigned int       checked_mul<unsigned int      >(unsigned int       lhs, unsigned int       rhs);
template unsigned long      checked_mul<unsigned long     >(unsigned long      lhs, unsigned long      rhs);
template unsigned long long checked_mul<unsigned long long>(unsigned long long lhs, unsigned long long rhs);

template<typename T>
T saturated_mul(T lhs, T rhs) NOEXCEPT {
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

template unsigned char      saturated_mul<unsigned char     >(unsigned char      lhs, unsigned char      rhs) NOEXCEPT;
template unsigned short     saturated_mul<unsigned short    >(unsigned short     lhs, unsigned short     rhs) NOEXCEPT;
template unsigned int       saturated_mul<unsigned int      >(unsigned int       lhs, unsigned int       rhs) NOEXCEPT;
template unsigned long      saturated_mul<unsigned long     >(unsigned long      lhs, unsigned long      rhs) NOEXCEPT;
template unsigned long long saturated_mul<unsigned long long>(unsigned long long lhs, unsigned long long rhs) NOEXCEPT;

}

#endif
