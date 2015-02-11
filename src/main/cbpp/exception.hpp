// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_CBPP_EXCEPTION_HPP_
#define POSEIDON_CBPP_EXCEPTION_HPP_

#include "../exception.hpp"
#include "status.hpp"

namespace Poseidon {

class CbppMessageException : public ProtocolException {
public:
	CbppMessageException(const char *file, std::size_t line,
		CbppStatus status, SharedNts message = SharedNts());
	~CbppMessageException() NOEXCEPT;

public:
	CbppStatus status() const NOEXCEPT {
		return CbppStatus(ProtocolException::code());
	}
};

}

#endif
