// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_MYSQL_EXCEPTION_HPP_
#define POSEIDON_MYSQL_EXCEPTION_HPP_

#include "../exception.hpp"

namespace Poseidon {

namespace MySql {
	class SqlException : public ProtocolException {
	public:
		SqlException(const char *file, std::size_t line, unsigned code, SharedNts message);
		~SqlException() NOEXCEPT;
	};
}

}

#endif
