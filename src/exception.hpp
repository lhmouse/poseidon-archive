// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2016, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_EXCEPTION_HPP_
#define POSEIDON_EXCEPTION_HPP_

#include "cxx_ver.hpp"
#include <exception>
#include <cstddef>
#include "log.hpp"
#include "shared_nts.hpp"

namespace Poseidon {

class Exception : public std::exception {
protected:
	const char *m_file;
	std::size_t m_line;
	const char *m_func;
	SharedNts m_message; // 拷贝构造函数不抛出异常。

public:
	Exception(const char *file, std::size_t line, const char *func, SharedNts message);
	~Exception() NOEXCEPT;

public:
	const char *what() const NOEXCEPT OVERRIDE {
		return m_message.get();
	}

	const char *get_file() const NOEXCEPT {
		return m_file;
	}
	std::size_t get_line() const NOEXCEPT {
		return m_line;
	}
	const char *get_func() const NOEXCEPT {
		return m_func;
	}
	const char *get_message() const NOEXCEPT {
		return m_message.get();
	}
};

typedef Exception BasicException;

}

#define DEBUG_THROW_IMPL(predictor_, etype_, parenthesis_, ...)	\
	do {	\
		if(predictor_){	\
			etype_ e_(__FILE__, __LINE__, __PRETTY_FUNCTION__, ## __VA_ARGS__);	\
			parenthesis_;	\
			throw e_;	\
		}	\
	} while(false)

#define DEBUG_THROW(etype_, ...)	\
	DEBUG_THROW_IMPL(true, etype_, {	\
		}, ## __VA_ARGS__)

#define DEBUG_THROW_UNLESS(expr_, etype_, ...)	\
	DEBUG_THROW_IMPL((expr_) ? false : true, etype_, {	\
		LOG_POSEIDON_ERROR("Pre-condition not met: ", #expr_);	\
		}, ## __VA_ARGS__)

#define DEBUG_THROW_ASSERT(expr_)	\
	DEBUG_THROW_IMPL((expr_) ? false : true, ::Poseidon::Exception, {	\
		LOG_POSEIDON_ERROR("Assertion failure: " #expr_);	\
		}, ::Poseidon::sslit("Assertion failure: " #expr_))

#endif
