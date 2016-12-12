// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2016, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_MYSQL_FORMATTING_HPP_
#define POSEIDON_MYSQL_FORMATTING_HPP_

#include <iosfwd>
#include <string>
#include <boost/cstdint.hpp>

namespace Poseidon {

class Uuid;

namespace MySql {
	struct StringEscaper {
		const std::string &str;

		explicit StringEscaper(const std::string &str_)
			: str(str_)
		{
		}
	};

	extern std::ostream &operator<<(std::ostream &os, const StringEscaper &rhs);

	struct DateTimeFormatter {
		const boost::uint64_t &time;

		explicit DateTimeFormatter(const boost::uint64_t &time_)
			: time(time_)
		{
		}
	};

	extern std::ostream &operator<<(std::ostream &os, const DateTimeFormatter &rhs);

	struct UuidFormatter {
		const Uuid &uuid;

		explicit UuidFormatter(const Uuid &uuid_)
			: uuid(uuid_)
		{
		}
	};

	extern std::ostream &operator<<(std::ostream &os, const UuidFormatter &rhs);
}

}

#endif
