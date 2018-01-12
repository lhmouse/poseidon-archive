// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_CBPP_EXCEPTION_HPP_
#define POSEIDON_CBPP_EXCEPTION_HPP_

#include "../exception.hpp"
#include "status_codes.hpp"

namespace Poseidon {
namespace Cbpp {

class Exception : public BasicException {
private:
	StatusCode m_status_code;

public:
	Exception(const char *file, std::size_t line, const char *func, StatusCode status_code, SharedNts message = SharedNts());
	~Exception() NOEXCEPT;

public:
	StatusCode get_status_code() const NOEXCEPT {
		return m_status_code;
	}
};

}
}

#endif
