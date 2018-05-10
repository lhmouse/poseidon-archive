// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_HTTP_EXCEPTION_HPP_
#define POSEIDON_HTTP_EXCEPTION_HPP_

#include "../exception.hpp"
#include "../optional_map.hpp"
#include "status_codes.hpp"

namespace Poseidon {
namespace Http {

extern const Optional_map &empty_headers() NOEXCEPT;

class Exception : public Basic_exception {
private:
	Status_code m_status_code;
	boost::shared_ptr<Optional_map> m_headers;

public:
	Exception(const char *file, std::size_t line, const char *func, Status_code status_code, Optional_map headers = Optional_map());
	~Exception() NOEXCEPT;

public:
	Status_code get_status_code() const NOEXCEPT {
		return m_status_code;
	}
	const Optional_map &get_headers() const NOEXCEPT {
		return m_headers ? *m_headers : empty_headers();
	}
};

}
}

#endif
