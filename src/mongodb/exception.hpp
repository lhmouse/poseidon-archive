// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2017, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_MONGODB_EXCEPTION_HPP_
#define POSEIDON_MONGODB_EXCEPTION_HPP_

#include "../protocol_exception.hpp"

namespace Poseidon {
namespace MongoDb {

class Exception : public ProtocolException {
private:
	SharedNts m_database;

public:
	Exception(const char *file, std::size_t line, const char *func, SharedNts database, unsigned long code, SharedNts message);
	~Exception() NOEXCEPT;

public:
	const char *get_database() const {
		return m_database.get();
	}
};

}
}

#endif
