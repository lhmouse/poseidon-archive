// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_MYSQL_EXCEPTION_HPP_
#define POSEIDON_MYSQL_EXCEPTION_HPP_

#include "../exception.hpp"

namespace Poseidon {
namespace Mysql {

class Exception : public Basic_exception {
private:
	Rcnts m_schema;
	unsigned long m_code;

public:
	Exception(const char *file, std::size_t line, const char *func, Rcnts schema, unsigned long code, Rcnts message);
	~Exception() NOEXCEPT;

public:
	const char * get_schema() const NOEXCEPT {
		return m_schema.get();
	}
	unsigned long get_code() const NOEXCEPT {
		return m_code;
	}
};

}
}

#endif
