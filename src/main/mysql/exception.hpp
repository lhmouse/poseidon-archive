// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_MYSQL_EXCEPTION_HPP_
#define POSEIDON_MYSQL_EXCEPTION_HPP_

#include "../exception.hpp"

namespace Poseidon {

class MySqlException : public ProtocolException {
public:
	MySqlException(const char *file, std::size_t line,
		unsigned code, const char *message);
	~MySqlException() NOEXCEPT;
};

}

#endif
