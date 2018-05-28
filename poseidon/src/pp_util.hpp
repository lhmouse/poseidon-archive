// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_PP_UTIL_HPP_
#define POSEIDON_PP_UTIL_HPP_

#define POSEIDON_STRINGIFY(...)        #__VA_ARGS__

#define POSEIDON_WIDEN(x_)             L##x_
#define POSEIDON_UTF8(x_)              u8##x_
#define POSEIDON_UTF16(x_)             u##x_
#define POSEIDON_UTF32(x_)             U##x_

#define POSEIDON_FIRST(x_, ...)        x_
#define POSEIDON_REST(x_, ...)         __VA_ARGS__

#define POSEIDON_CAT2(x_, y_)          x_##y_
#define POSEIDON_CAT3(x_, y_, z_)      x_##y_##z_

#define POSEIDON_LAZY(f_, ...)         f_(__VA_ARGS__)

#define POSEIDON_UNIQUE_NAME           POSEIDON_LAZY(POSEIDON_CAT3, poseidon_unique_, __COUNTER__, _Ztq_)

#endif
