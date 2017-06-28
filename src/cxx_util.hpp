// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2017, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_CXX_UTIL_HPP_
#define POSEIDON_CXX_UTIL_HPP_

#define TOKEN_TO_STR_(...)          # __VA_ARGS__
#define TOKEN_TO_STR(...)           TOKEN_TO_STR_(__VA_ARGS__)

#define STRIP_FIRST_(x_, ...)       __VA_ARGS__
#define STRIP_FIRST(...)            STRIP_FIRST_(__VA_ARGS__)

#define TOKEN_CAT2_(x_, y_)         x_ ## y_
#define TOKEN_CAT2(x_, y_)          TOKEN_CAT2_(x_, y_)

#define TOKEN_CAT3_(x_, y_, z_)     x_ ## y_ ## z_
#define TOKEN_CAT3(x_, y_, z_)      TOKEN_CAT3_(x_, y_, z_)

#define MAGIC_LN_2_(ln_)            Poseidon_magic_ ## ln_ ## X_
#define MAGIC_LN_1_(ln_)            MAGIC_LN_2_(ln_)

#define UNIQUE_ID                   MAGIC_LN_1_(__COUNTER__)

#include <boost/range/begin.hpp>
#include <boost/range/end.hpp>
#include <boost/range/size.hpp>
#include <cstddef>

#define BEGIN(ar_)                  (::boost::begin(ar_))
#define END(ar_)                    (::boost::end(ar_))
#define COUNT_OF(ar_)               (static_cast< ::std::size_t>(::boost::size(ar_)))

namespace Poseidon {

template<unsigned long UniqueT>
class Noncopyable {
public:
#if __cplusplus >= 201103l
	Noncopyable() = default;
#else
	Noncopyable(){ }
#endif

private:
	Noncopyable(const Noncopyable &);
	Noncopyable &operator=(const Noncopyable &);
};

}

#define NONCOPYABLE                 private ::Poseidon::Noncopyable<__COUNTER__>

#endif
