// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2016, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_MYSQL_EXCEPTION_HPP_
#define POSEIDON_MYSQL_EXCEPTION_HPP_

#include "../protocol_exception.hpp"

namespace Poseidon {

namespace MySql {
	class Exception : public ProtocolException {
	private:
		SharedNts m_schema;

	public:
		Exception(const char *file, std::size_t line, const char *func, SharedNts schema, long code, SharedNts message);
		~Exception() NOEXCEPT;

	public:
		const char *get_schema() const {
			return m_schema.get();
		}
	};
}

}

#endif
