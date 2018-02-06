// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_ENDIAN_HPP_
#define POSEIDON_ENDIAN_HPP_

#include "cxx_util.hpp"
#include <boost/cstdint.hpp>
#include <boost/type_traits/is_integral.hpp>
#include <boost/type_traits/make_unsigned.hpp>
#include <boost/type_traits/common_type.hpp>
#include <boost/static_assert.hpp>

#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
#  define POSEIDON_BSWAP_UNLESS_BE(bits_)   //
#  define POSEIDON_BSWAP_UNLESS_LE(bits_)   __builtin_bswap##bits_
#elif __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
#  define POSEIDON_BSWAP_UNLESS_BE(bits_)   __builtin_bswap##bits_
#  define POSEIDON_BSWAP_UNLESS_LE(bits_)   //
#else
#  error This architecture is not supported.
#endif

namespace Poseidon {

template<unsigned kSizeT>
struct ByteSwapper;

template<>
struct ByteSwapper<1> {
	static boost::uint8_t bswap_unless_be(boost::uint8_t value){
		return value;
	}
	static boost::uint8_t bswap_unless_le(boost::uint8_t value){
		return value;
	}
};

template<>
struct ByteSwapper<2> {
	static boost::uint16_t bswap_unless_be(boost::uint16_t value){
		return POSEIDON_BSWAP_UNLESS_BE(16)(value);
	}
	static boost::uint16_t bswap_unless_le(boost::uint16_t value){
		return POSEIDON_BSWAP_UNLESS_LE(16)(value);
	}
};

template<>
struct ByteSwapper<4> {
	static boost::uint32_t bswap_unless_be(boost::uint32_t value){
		return POSEIDON_BSWAP_UNLESS_BE(32)(value);
	}
	static boost::uint32_t bswap_unless_le(boost::uint32_t value){
		return POSEIDON_BSWAP_UNLESS_LE(32)(value);
	}
};

template<>
struct ByteSwapper<8> {
	static boost::uint64_t bswap_unless_be(boost::uint64_t value){
		return POSEIDON_BSWAP_UNLESS_BE(64)(value);
	}
	static boost::uint64_t bswap_unless_le(boost::uint64_t value){
		return POSEIDON_BSWAP_UNLESS_LE(64)(value);
	}
};

template<typename ValueT>
ValueT load_be(ValueT value){
	BOOST_STATIC_ASSERT(boost::is_integral<ValueT>::value);
	const AUTO(temp, static_cast<typename boost::make_unsigned<ValueT>::type>(value));
	return static_cast<ValueT>(ByteSwapper<sizeof(temp)>::bswap_unless_be(temp));
}
template<typename ValueT>
void store_be(ValueT &ref, typename boost::common_type<ValueT>::type value){
	BOOST_STATIC_ASSERT(boost::is_integral<ValueT>::value);
	const AUTO(temp, static_cast<typename boost::make_unsigned<ValueT>::type>(value));
	ref = static_cast<ValueT>(ByteSwapper<sizeof(temp)>::bswap_unless_be(temp));
}

template<typename ValueT>
ValueT load_le(ValueT value){
	BOOST_STATIC_ASSERT(boost::is_integral<ValueT>::value);
	const AUTO(temp, static_cast<typename boost::make_unsigned<ValueT>::type>(value));
	return static_cast<ValueT>(ByteSwapper<sizeof(temp)>::bswap_unless_le(temp));
}
template<typename ValueT>
void store_le(ValueT &ref, typename boost::common_type<ValueT>::type value){
	BOOST_STATIC_ASSERT(boost::is_integral<ValueT>::value);
	const AUTO(temp, static_cast<typename boost::make_unsigned<ValueT>::type>(value));
	ref = static_cast<ValueT>(ByteSwapper<sizeof(temp)>::bswap_unless_le(temp));
}

}

#endif
