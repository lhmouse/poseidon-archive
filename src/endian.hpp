// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2017, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_ENDIAN_HPP_
#define POSEIDON_ENDIAN_HPP_

#include "cxx_util.hpp"
#include <climits>
#include <boost/type_traits/make_unsigned.hpp>
#include <boost/type_traits/is_integral.hpp>
#include <boost/static_assert.hpp>

namespace Poseidon {

template<typename ValueT>
inline ValueT load_le(const ValueT &mem){
	BOOST_STATIC_ASSERT(boost::is_integral<ValueT>::value);
	typedef typename boost::make_unsigned<ValueT>::type Unsigned;

	const unsigned char *read = reinterpret_cast<const unsigned char *>(&mem);
	Unsigned u = 0;
	for(unsigned i = 0; i < sizeof(u); ++i){
		u |= static_cast<Unsigned>(*(read++)) << (i * CHAR_BIT);
	}
	return static_cast<ValueT>(u);
}
template<typename ValueT>
inline ValueT load_be(const ValueT &mem){
	BOOST_STATIC_ASSERT(boost::is_integral<ValueT>::value);
	typedef typename boost::make_unsigned<ValueT>::type Unsigned;

	const unsigned char *read = reinterpret_cast<const unsigned char *>(&mem);
	Unsigned u = 0;
	for(unsigned i = sizeof(u); i > 0; --i){
		u |= static_cast<Unsigned>(*(read++)) << ((i - 1) * CHAR_BIT);
	}
	return static_cast<ValueT>(u);
}

template<typename ValueT>
inline void store_le(ValueT &mem, typename Identity<ValueT>::type val){
	BOOST_STATIC_ASSERT(boost::is_integral<ValueT>::value);
	typedef typename boost::make_unsigned<ValueT>::type Unsigned;

	unsigned char *write = reinterpret_cast<unsigned char *>(&mem);
	Unsigned u = static_cast<Unsigned>(val);
	for(unsigned i = 0; i < sizeof(u); ++i){
		*(write++) = static_cast<unsigned char>(u >> (i * CHAR_BIT));
	}
}
template<typename ValueT>
inline void store_be(ValueT &mem, typename Identity<ValueT>::type val){
	BOOST_STATIC_ASSERT(boost::is_integral<ValueT>::value);
	typedef typename boost::make_unsigned<ValueT>::type Unsigned;

	unsigned char *write = reinterpret_cast<unsigned char *>(&mem);
	Unsigned u = static_cast<Unsigned>(val);
	for(unsigned i = sizeof(u); i > 0; --i){
		*(write++) = static_cast<unsigned char>(u >> ((i - 1) * CHAR_BIT));
	}
}

}

#endif
