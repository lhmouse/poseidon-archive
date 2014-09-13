#ifndef POSEIDON_CXX_VER_HPP_
#define POSEIDON_CXX_VER_HPP_

#include <utility>
#include <memory>
#include <cstddef>
#include <boost/type_traits/remove_cv.hpp>
#include <boost/type_traits/remove_reference.hpp>
#include <boost/scoped_ptr.hpp>
#include <boost/scoped_array.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/weak_ptr.hpp>

#if __cplusplus >= 201103l
#	define POSEIDON_CXX11
#endif

#if __cplusplus >= 201402l
#	define POSEIDON_CXX14
#endif

#ifdef POSEIDON_CXX11
#	define DECLTYPE(expr_)			decltype(expr_)
#	define CONSTEXPR				constexpr
#	define NOEXCEPT					noexcept
#else
#	define DECLTYPE(expr_)			__typeof__(expr_)
#	define CONSTEXPR
#	define NOEXCEPT					throw()
#endif

namespace Poseidon {

template<typename Type>
typename boost::remove_cv<
	typename boost::remove_reference<Type>::type
	>::type
	valueOfHelper_(const Type &) NOEXCEPT;

struct Nullptr_t_ {
#ifdef POSEIDON_CXX11
	explicit
#endif
	CONSTEXPR operator bool() const NOEXCEPT {
		return false;
	}

	template<typename T>
	CONSTEXPR operator T *() const NOEXCEPT {
		return 0;
	}
	template<typename C, typename M>
	CONSTEXPR operator C M::*() const NOEXCEPT {
		return 0;
	}
#ifdef POSEIDON_CXX11
	CONSTEXPR operator std::nullptr_t() const noexcept {
		return nullptr;
	}
#endif

	template<typename T>
	operator std::auto_ptr<T>() const NOEXCEPT {
		return std::auto_ptr<T>();
	}
#ifdef POSEIDON_CXX11
	template<typename T, typename D>
	operator std::unique_ptr<T, D>() const NOEXCEPT {
		return std::unique_ptr<T, D>();
	}
	template<typename T>
	operator std::shared_ptr<T>() const NOEXCEPT {
		return std::shared_ptr<T>();
	}
	template<typename T>
	operator std::weak_ptr<T>() const NOEXCEPT {
		return std::weak_ptr<T>();
	}
#endif

	template<typename T>
	operator boost::scoped_ptr<T>() const NOEXCEPT {
		return boost::scoped_ptr<T>();
	}
	template<typename T>
	operator boost::scoped_array<T>() const NOEXCEPT {
		return boost::scoped_array<T>();
	}
	template<typename T>
	operator boost::shared_ptr<T>() const NOEXCEPT {
		return boost::shared_ptr<T>();
	}
	template<typename T>
	operator boost::weak_ptr<T>() const NOEXCEPT {
		return boost::weak_ptr<T>();
	}

private:
	void *unused_;

	void operator&() const;
};

template<typename Type>
typename boost::add_reference<Type>::type declRef_() NOEXCEPT;

}

#ifdef POSEIDON_CXX11
#	define AUTO(id_, init_)			auto id_ = init_
#	define AUTO_REF(id_, init_)		auto &id_ = init_
#	define NULLPTR					(::Poseidon::Nullptr_t_())	// (nullptr)
#	define STD_MOVE(expr_)			(::std::move(expr_))
#	define STD_FORWARD(t_, expr_)	(::std::forward<t_>(expr_))
#	define DECLREF(t_)				(::std::declval<typename ::std::remove_reference<t_>::type>())
#else
#	define AUTO(id_, init_)			DECLTYPE(::Poseidon::valueOfHelper_(init_)) id_(init_)
#	define AUTO_REF(id_, init_)		DECLTYPE(init_) &id_ = (init_)
#	define NULLPTR					(::Poseidon::Nullptr_t_())
#	define STD_MOVE(expr_)			(expr_)
#	define STD_FORWARD(t_, expr_)	(expr_)
#	define DECLREF(t_)				(::Poseidon::declRef_<t_>())
#endif

#endif
