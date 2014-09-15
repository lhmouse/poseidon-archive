#ifndef POSEIDON_CXX_UTIL_HPP_
#define POSEIDON_CXX_UTIL_HPP_

#include <cstddef>

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

#define COUNT_OF(ar)		sizeof(::Poseidon::countOfHelper(ar))
#define ARRAY_BEGIN(ar)		(::Poseidon::arrayBegin(ar))
#define ARRAY_END(ar)		(::Poseidon::arrayEnd(ar))

#define MAGIC_LN_2_(ln_)	Poseidon_magic_ ## ln_ ## _
#define MAGIC_LN_1_(ln_)	MAGIC_LN_2_(ln_)
#define UNIQUE_ID			MAGIC_LN_1_(__LINE__)

#endif
