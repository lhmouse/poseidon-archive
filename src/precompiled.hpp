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
#include <boost/type_traits/remove_cv.hpp>
#include <boost/type_traits/remove_reference.hpp>

namespace Poseidon {

template<typename Type>
typename boost::remove_cv<
	typename boost::remove_reference<Type>::type
	>::type
	valueOfHelper(const Type &);

template<typename Type, std::size_t COUNT>
char (&countOfHelper(const Type (&)[COUNT]))[COUNT];

}

#define DECLTYPE(expr)			__typeof__(expr)
#define AUTO(id, init)			DECLTYPE(::Poseidon::valueOfHelper(init)) id = (init)
#define AUTO_REF(id, init)		DECLTYPE(init) &id = (init)

#define COUNT_OF(ar)			sizeof(::Poseidon::countOfHelper(ar))

#endif
