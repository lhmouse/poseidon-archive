#ifndef POSEIDON_ENDIAN_HPP_
#define POSEIDON_ENDIAN_HPP_

#include <climits>
#include <boost/type_traits/make_unsigned.hpp>
#include <boost/type_traits/is_integral.hpp>
#include <boost/type_traits/common_type.hpp>
#include <boost/static_assert.hpp>

namespace Poseidon {

template<typename ValueT>
ValueT loadLe(const ValueT &mem){
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
ValueT loadBe(const ValueT &mem){
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
void storeLe(ValueT &mem, typename boost::common_type<ValueT>::type val){
	BOOST_STATIC_ASSERT(boost::is_integral<ValueT>::value);
	typedef typename boost::make_unsigned<ValueT>::type Unsigned;

	unsigned char *write = reinterpret_cast<unsigned char *>(&mem);
	Unsigned u = static_cast<Unsigned>(val);
	for(unsigned i = 0; i < sizeof(u); ++i){
		*(write++) = static_cast<unsigned char>(u >> (i * CHAR_BIT));
	}
}
template<typename ValueT>
void storeBe(ValueT &mem, typename boost::common_type<ValueT>::type val){
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
