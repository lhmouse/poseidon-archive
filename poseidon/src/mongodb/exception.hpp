// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_MONGODB_EXCEPTION_HPP_
#define POSEIDON_MONGODB_EXCEPTION_HPP_

#include "../exception.hpp"

namespace Poseidon {
namespace Mongodb {

class Exception : public Basic_exception {
private:
	Rcnts m_database;
	unsigned long m_code;

public:
	Exception(const char *file, std::size_t line, const char *func, Rcnts database, unsigned long code, Rcnts message);
	~Exception() NOEXCEPT;

public:
	const char *get_database() const NOEXCEPT {
		return m_database.get();
	}
	unsigned long get_code() const NOEXCEPT {
		return m_code;
	}
};

}
}

#endif
