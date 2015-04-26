// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_FLAGS_HPP_
#define POSEIDON_FLAGS_HPP_

#include "cxx_util.hpp"

namespace Poseidon {

template<typename T>
T &addFlags(T &val, typename Identity<T>::type flags){
	val |= flags;
	return val;
}
template<typename T>
T &removeFlags(T &val, typename Identity<T>::type flags){
	val &= ~flags;
	return val;
}

template<typename T>
bool hasAllFlagsOf(const T &val, typename Identity<T>::type flags){
	return (val & flags) == flags;
}
template<typename T>
bool hasAnyFlagsOf(const T &val, typename Identity<T>::type flags){
	return (val & flags) != 0;
}
template<typename T>
bool hasNoneFlagsOf(const T &val, typename Identity<T>::type flags){
	return (val & flags) == 0;
}

}

#endif
