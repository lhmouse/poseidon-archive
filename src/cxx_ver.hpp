// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_CXX_VER_HPP_
#define POSEIDON_CXX_VER_HPP_

#include <utility>
#include <memory>
#include <cstddef>

#if __cplusplus >= 201103l
#   define POSEIDON_CXX11
#endif

#if __cplusplus >= 201402l
#   define POSEIDON_CXX14
#endif

#ifdef POSEIDON_CXX11
#   define ENABLE_IF_CXX11(...)     __VA_ARGS__
#else
#   define ENABLE_IF_CXX11(...)
#endif

#ifdef POSEIDON_CXX14
#   define ENABLE_IF_CXX14(...)     __VA_ARGS__
#else
#   define ENABLE_IF_CXX14(...)
#endif

#ifdef POSEIDON_CXX11
#   include <type_traits>
#else
#   include <boost/type_traits/add_reference.hpp>
#   include <boost/type_traits/remove_cv.hpp>
#   include <boost/type_traits/remove_reference.hpp>
#   include <boost/type_traits/decay.hpp>
#endif

#ifdef POSEIDON_CXX11
#   define CONSTEXPR                constexpr
#   define NOEXCEPT                 noexcept
#   define OVERRIDE                 override
#   define FINAL                    final
#else
#   define CONSTEXPR
#   define NOEXCEPT                 throw()
#   define OVERRIDE
#   define FINAL
#endif

namespace Poseidon {

#ifdef POSEIDON_CXX11
template<typename T>
typename std::remove_cv<
	typename std::remove_reference<
		typename std::decay<T>::type
		>::type
	>::type value_of_helper(const T &);
#else
template<typename T>
typename boost::remove_cv<
	typename boost::remove_reference<
		typename boost::decay<T>::type
		>::type
	>::type value_of_helper(const T &);
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
		return ret; // RVO
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

// 对 C++98 中无法移动构造 std::tr1::function / boost::function 的补偿。
// 只能用于直接初始化对象。
template<typename T>
T move_as_identity(T &rhs) NOEXCEPT {
	return Move<T>(rhs); // 隐式转换，无视构造函数。
}
template<typename T>
T move_as_identity(Move<T> rhs) NOEXCEPT {
	return rhs; // 隐式转换，无视构造函数。
}
#endif

#ifndef POSEIDON_CXX11
template<typename T>
typename boost::add_reference<T>::type decl_ref() NOEXCEPT;

struct ValueInitializer {
	template<typename T>
	operator T() const {
		return T();
	}
};
#endif

}

#ifdef POSEIDON_CXX11
#   define CV_VALUE_TYPE(...)       typename ::std::remove_reference<decltype(__VA_ARGS__)>::type
#   define VALUE_TYPE(...)          decltype(::Poseidon::value_of_helper(__VA_ARGS__))
#   define AUTO(id_, ...)           auto id_ = __VA_ARGS__
#   define AUTO_REF(id_, ...)       auto &id_ = __VA_ARGS__
#   define STD_MOVE(expr_)          (::std::move(expr_))
#   define STD_MOVE_IDN(expr_)      (::std::move(expr_))
#   define DECLREF(t_)              (::std::declval<typename ::std::add_lvalue_reference<t_>::type>())
#   define VAL_INIT                 { }
#   define NULLPTR                  nullptr
#else
#   define CV_VALUE_TYPE(...)       __typeof__(__VA_ARGS__)
#   define VALUE_TYPE(...)          __typeof__(::Poseidon::value_of_helper(__VA_ARGS__))
#   define AUTO(id_, ...)           VALUE_TYPE(__VA_ARGS__) id_(__VA_ARGS__)
#   define AUTO_REF(id_, ...)       CV_VALUE_TYPE(__VA_ARGS__) &id_ = (__VA_ARGS__)
#   define STD_MOVE(expr_)          (::Poseidon::move(expr_))
#   define STD_MOVE_IDN(expr_)      (::Poseidon::move_as_identity(expr_))
#   define DECLREF(t_)              (::Poseidon::decl_ref<t_>())
#   define VAL_INIT                 (::Poseidon::ValueInitializer())
#   define NULLPTR                  VAL_INIT
#endif

#endif
