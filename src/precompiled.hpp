#ifndef POSEIDON_PRECOMPILED_HPP_
#define POSEIDON_PRECOMPILED_HPP_

#include <vector>
#include <list>
#include <map>
#include <set>
#include <string>
#include <iterator>
#include <algorithm>
#include <utility>
#include <sstream>

#include <cassert>
#include <cstring>
//#include <cstdlib> // 使用 <boost/cstdlib.hpp>。
#include <cstddef>
#include <cstdio>

#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>
#include <boost/scoped_ptr.hpp>
#include <boost/scoped_array.hpp>
#include <boost/enable_shared_from_this.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/function.hpp>
#include <boost/cstdlib.hpp>
#include <boost/cstdint.hpp>

namespace Poseidon {

template<typename Type, std::size_t COUNT>
char (&countOfHelper(const Type (&)[COUNT]))[COUNT];

template<typename Type, std::size_t COUNT>
Type *arrayBegin(Type (&array)[COUNT]){
	return array;
}
template<typename Type, std::size_t COUNT>
Type *arrayEnd(Type (&array)[COUNT]){
	return array + COUNT;
}

}

#define COUNT_OF(ar)			sizeof(::Poseidon::countOfHelper(ar))
#define ARRAY_BEGIN(ar)			(::Poseidon::arrayBegin(ar))
#define ARRAY_END(ar)			(::Poseidon::arrayEnd(ar))

#include "cxxver.hpp"

#endif
