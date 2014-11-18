// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_CXX_VER_HPP_
#define POSEIDON_CXX_VER_HPP_

#include <utility>
#include <memory>
#include <cstddef>

#if __cplusplus >= 201103l
#	define POSEIDON_CXX11
#endif

#if __cplusplus >= 201402l
#	define POSEIDON_CXX14
#endif

#ifdef POSEIDON_CXX11
#	include <type_traits>
#else
#	include <boost/type_traits/add_reference.hpp>
#	include <boost/type_traits/remove_cv.hpp>
#	include <boost/type_traits/remove_reference.hpp>
#	include <boost/type_traits/decay.hpp>
#endif

#ifdef POSEIDON_CXX11
#	define CONSTEXPR				constexpr
#	define NOEXCEPT					noexcept
#else
#	define CONSTEXPR
#	define NOEXCEPT					throw()
#endif

namespace Poseidon {

#ifdef POSEIDON_CXX11
template<typename T>
typename std::remove_cv<
	typename std::remove_reference<
		typename std::decay<T>::type
		>::type
	>::type valueOfHelper(const T &);
#else
template<typename T>
typename boost::remove_cv<
	typename boost::remove_reference<
		typename boost::decay<T>::type
		>::type
	>::type valueOfHelper(const T &);
#endif

#ifdef POSEIDON_CXX11
template<typename T>
using Move = typename std::remove_reference<T>::type &&;
#else
template<typename T>
class Move {
private:
	T &m_holds;

public:
	explicit Move(T &rhs) NOEXCEPT
		: m_holds(rhs)
	{
	}
	Move(const Move &rhs) NOEXCEPT
		: m_holds(rhs.m_holds)
	{
	}

public:
	void swap(T &rhs){
		using std::swap;
		swap(m_holds, rhs);
	}
	void swap(Move &rhs){
		swap(rhs.m_holds);
	}

public:
	operator T() const {
		T ret;
		using std::swap;
		swap(ret, m_holds);
		return ret;	// RVO
	}
};

template<typename T>
void swap(Move<T> &lhs, Move<T> &rhs){
	lhs.swap(rhs);
}
template<typename T>
void swap(Move<T> &lhs, T &rhs){
	lhs.swap(rhs);
}
template<typename T>
void swap(T &lhs, Move<T> &rhs){
	rhs.swap(lhs);
}

template<typename T>
Move<T> move(T &rhs) NOEXCEPT {
	return Move<T>(rhs);
}
template<typename T>
Move<T> move(Move<T> rhs) NOEXCEPT {
	return rhs;
}
#endif

#ifndef POSEIDON_CXX11
template<typename T>
typename boost::add_reference<T>::type declRef() NOEXCEPT;

struct ValueInitializer {
	template<typename T>
	operator T() const {
		return T();
	}
};
#endif

}

#ifdef POSEIDON_CXX11
#	define CV_VALUE_TYPE(expr_)		typename ::std::remove_reference<decltype(expr_)>::type
#	define VALUE_TYPE(expr_)		decltype(::Poseidon::valueOfHelper(expr_))
#	define AUTO(id_, init_)			auto id_ = init_
#	define AUTO_REF(id_, init_)		auto &id_ = init_
#	define STD_MOVE(expr_)			(::std::move(expr_))
#	define DECLREF(t_)				(::std::declval<typename ::std::add_lvalue_reference<t_>::type>())
#	define VAL_INIT					{ }
#else
#	define CV_VALUE_TYPE(expr_)		__typeof__(expr_)
#	define VALUE_TYPE(expr_)		__typeof__(::Poseidon::valueOfHelper(expr_))
#	define AUTO(id_, init_)			VALUE_TYPE(init_) id_(init_)
#	define AUTO_REF(id_, init_)		CV_VALUE_TYPE(init_) &id_ = (init_)
#	define STD_MOVE(expr_)			(::Poseidon::move(expr_))
#	define DECLREF(t_)				(::Poseidon::declRef<t_>())
#	define VAL_INIT					(::Poseidon::ValueInitializer())
#endif

#endif
