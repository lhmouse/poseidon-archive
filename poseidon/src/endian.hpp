// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_ENDIAN_HPP_
#define POSEIDON_ENDIAN_HPP_

#include "cxx_ver.hpp"
#include <boost/cstdint.hpp>
#include <boost/type_traits/is_integral.hpp>
#include <boost/type_traits/make_unsigned.hpp>
#include <boost/type_traits/common_type.hpp>
#include <boost/static_assert.hpp>
#include <byteswap.h>

#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
#  define POSEIDON_BSWAP_UNLESS_BE(bits_)   //
#  define POSEIDON_BSWAP_UNLESS_LE(bits_)   __bswap_##bits_
#elif __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
#  define POSEIDON_BSWAP_UNLESS_BE(bits_)   __bswap_##bits_
#  define POSEIDON_BSWAP_UNLESS_LE(bits_)   //
#else
#  error This architecture is not supported.
#endif

namespace Poseidon {

template<std::size_t sizeT>
struct Byte_swapper;

template<>
struct Byte_swapper<1> {
	static std::uint8_t bswap_unless_be(std::uint8_t value){
		return value;
	}
	static std::uint8_t bswap_unless_le(std::uint8_t value){
		return value;
	}
};

template<>
struct Byte_swapper<2> {
	static std::uint16_t bswap_unless_be(std::uint16_t value){
		return POSEIDON_BSWAP_UNLESS_BE(16)(value);
	}
	static std::uint16_t bswap_unless_le(std::uint16_t value){
		return POSEIDON_BSWAP_UNLESS_LE(16)(value);
	}
};

template<>
struct Byte_swapper<4> {
	static std::uint32_t bswap_unless_be(std::uint32_t value){
		return POSEIDON_BSWAP_UNLESS_BE(32)(value);
	}
	static std::uint32_t bswap_unless_le(std::uint32_t value){
		return POSEIDON_BSWAP_UNLESS_LE(32)(value);
	}
};

template<>
struct Byte_swapper<8> {
	static std::uint64_t bswap_unless_be(std::uint64_t value){
		return POSEIDON_BSWAP_UNLESS_BE(64)(value);
	}
	static std::uint64_t bswap_unless_le(std::uint64_t value){
		return POSEIDON_BSWAP_UNLESS_LE(64)(value);
	}
};

template<typename ValueT>
ValueT load_be(ValueT value){
	BOOST_STATIC_ASSERT(boost::is_integral<ValueT>::value);
	const AUTO(temp, static_cast<typename boost::make_unsigned<ValueT>::type>(value));
	return static_cast<ValueT>(Byte_swapper<sizeof(temp)>::bswap_unless_be(temp));
}
template<typename ValueT>
void store_be(ValueT &ref, typename boost::common_type<ValueT>::type value){
	BOOST_STATIC_ASSERT(boost::is_integral<ValueT>::value);
	const AUTO(temp, static_cast<typename boost::make_unsigned<ValueT>::type>(value));
	ref = static_cast<ValueT>(Byte_swapper<sizeof(temp)>::bswap_unless_be(temp));
}

template<typename ValueT>
ValueT load_le(ValueT value){
	BOOST_STATIC_ASSERT(boost::is_integral<ValueT>::value);
	const AUTO(temp, static_cast<typename boost::make_unsigned<ValueT>::type>(value));
	return static_cast<ValueT>(Byte_swapper<sizeof(temp)>::bswap_unless_le(temp));
}
template<typename ValueT>
void store_le(ValueT &ref, typename boost::common_type<ValueT>::type value){
	BOOST_STATIC_ASSERT(boost::is_integral<ValueT>::value);
	const AUTO(temp, static_cast<typename boost::make_unsigned<ValueT>::type>(value));
	ref = static_cast<ValueT>(Byte_swapper<sizeof(temp)>::bswap_unless_le(temp));
}

}

#endif
