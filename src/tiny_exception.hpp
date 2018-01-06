// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_TINY_EXCEPTION_HPP_
#define POSEIDON_TINY_EXCEPTION_HPP_

#include <exception>

namespace Poseidon {

class TinyException : public std::exception {
private:
	const char *m_static_msg;

public:
	explicit TinyException(const char *static_msg) throw()
		: m_static_msg(static_msg)
	{ }
	~TinyException() throw();

public:
	const char *what() const throw() {
		return m_static_msg;
	}
};

}

#endif
