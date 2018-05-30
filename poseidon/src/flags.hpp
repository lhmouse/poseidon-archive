// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_FLAGS_HPP_
#define POSEIDON_FLAGS_HPP_

#include <type_traits>
#include <boost/static_assert.hpp>

namespace Poseidon {

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wconversion"
#pragma GCC diagnostic ignored "-Wsign-conversion"

template<typename T>
inline T & add_flags(T &val, typename std::common_type<T>::type flags){
	BOOST_STATIC_ASSERT((std::is_enum<T>::value || std::is_integral<T>::value));

	val |= flags;
	return val;
}
template<typename T>
inline T & remove_flags(T &val, typename std::common_type<T>::type flags){
	BOOST_STATIC_ASSERT((std::is_enum<T>::value || std::is_integral<T>::value));

	val &= ~flags;
	return val;
}
template<typename T>
inline T & flip_flags(T &val, typename std::common_type<T>::type flags){
	BOOST_STATIC_ASSERT((std::is_enum<T>::value || std::is_integral<T>::value));

	val ^= flags;
	return val;
}

template<typename T>
inline bool has_all_flags_of(const T &val, typename std::common_type<T>::type flags){
	BOOST_STATIC_ASSERT((std::is_enum<T>::value || std::is_integral<T>::value));

	return (val & flags) == flags;
}
template<typename T>
inline bool has_any_flags_of(const T &val, typename std::common_type<T>::type flags){
	BOOST_STATIC_ASSERT((std::is_enum<T>::value || std::is_integral<T>::value));

	return (val & flags) != 0;
}
template<typename T>
inline bool has_none_flags_of(const T &val, typename std::common_type<T>::type flags){
	BOOST_STATIC_ASSERT((std::is_enum<T>::value || std::is_integral<T>::value));

	return (val & flags) == 0;
}

#pragma GCC diagnostic pop

}

#endif
