#ifndef POSEIDON_CXX_VER_HPP_
#define POSEIDON_CXX_VER_HPP_

#include <utility>
#include <boost/type_traits/remove_cv.hpp>
#include <boost/type_traits/remove_reference.hpp>

#if __cplusplus >= 201103l
#	define POSEIDON_CXX11
#endif

#if __cplusplus >= 201402l
#	define POSEIDON_CXX14
#endif

namespace Poseidon {

template<typename Type>
typename boost::remove_cv<
	typename boost::remove_reference<Type>::type
	>::type
	valueOfHelper_(const Type &);

}

#ifdef POSEIDON_CXX11
#	define DECLTYPE(expr_)			decltype(expr_)
#	define AUTO(id_, init_)			auto id_ = init_
#	define AUTO_REF(id_, init_)		auto &id_ = init_
#	define STD_MOVE(expr_)			::std::move(expr_)
#	define STD_FORWARD(t_, expr_)	::std::forward<t_>(expr_)
#else
#	define DECLTYPE(expr_)			__typeof__(expr_)
#	define AUTO(id_, init_)			DECLTYPE(::Poseidon::valueOfHelper_(init_)) id_(init_)
#	define AUTO_REF(id_, init_)		DECLTYPE(init_) &id_ = (init_)
#	define STD_MOVE(expr_)			(expr_)
#	define STD_FORWARD(t_, expr_)	(expr_)
#endif

#endif
