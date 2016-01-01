// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2016, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_FLAGS_HPP_
#define POSEIDON_FLAGS_HPP_

#include "cxx_util.hpp"

namespace Poseidon {

template<typename T>
T &add_flags(T &val, typename Identity<T>::type flags){
	val |= flags;
	return val;
}
template<typename T>
T &remove_flags(T &val, typename Identity<T>::type flags){
	val &= ~flags;
	return val;
}

template<typename T>
bool has_all_flags_of(const T &val, typename Identity<T>::type flags){
	return (val & flags) == flags;
}
template<typename T>
bool has_any_flags_of(const T &val, typename Identity<T>::type flags){
	return (val & flags) != 0;
}
template<typename T>
bool has_none_flags_of(const T &val, typename Identity<T>::type flags){
	return (val & flags) == 0;
}

}

#endif
