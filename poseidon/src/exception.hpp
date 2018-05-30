// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_EXCEPTION_HPP_
#define POSEIDON_EXCEPTION_HPP_

#include "cxx_ver.hpp"
#include <exception>
#include <cstddef>
#include "log.hpp"
#include "rcnts.hpp"

namespace Poseidon {

class Exception : public std::exception {
protected:
	const char *m_file;
	std::size_t m_line;
	const char *m_func;
	Rcnts m_message; // 拷贝构造函数不抛出异常。

public:
	Exception(const char *file, std::size_t line, const char *func, Rcnts message);
	~Exception() NOEXCEPT;

public:
	const char * what() const NOEXCEPT OVERRIDE;

	const char * get_file() const NOEXCEPT {
		return m_file;
	}
	std::size_t get_line() const NOEXCEPT {
		return m_line;
	}
	const char * get_func() const NOEXCEPT {
		return m_func;
	}
	const char * get_message() const NOEXCEPT {
		return m_message.get();
	}
};

typedef Exception Basic_exception;

}

#define POSEIDON_THROW_UNLESS_IMPL_(predictor_, etype_, parenthesis_, ...)	\
	do {	\
		if(predictor_){	\
			break;	\
		}	\
		etype_ e_(__FILE__, __LINE__, __PRETTY_FUNCTION__, ## __VA_ARGS__);	\
		{ parenthesis_ }	\
		throw e_;	\
	} while(false)

#define POSEIDON_THROW(etype_, ...)	\
	POSEIDON_THROW_UNLESS_IMPL_(false,	\
		etype_, { }, ## __VA_ARGS__)

#define POSEIDON_THROW_UNLESS(expr_, etype_, ...)	\
	POSEIDON_THROW_UNLESS_IMPL_((expr_),	\
		etype_, { POSEIDON_LOG_ERROR("Pre-condition not met: ", #expr_); }, ## __VA_ARGS__)

#define POSEIDON_THROW_ASSERT(expr_)	\
	POSEIDON_THROW_UNLESS_IMPL_((expr_),	\
		::Poseidon::Exception, { POSEIDON_LOG_ERROR("Assertion failure: " #expr_); }, ::Poseidon::Rcnts::view("Assertion failure: " #expr_))

#endif
