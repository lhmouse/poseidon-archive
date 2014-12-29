// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_MYSQL_UTILITIES_HPP_
#define POSEIDON_MYSQL_UTILITIES_HPP_

#include <iosfwd>
#include <string>
#include <boost/cstdint.hpp>

namespace Poseidon {

struct MySqlStringEscaper {
	const std::string &str;

	explicit MySqlStringEscaper(const std::string &str_)
		: str(str_)
	{
	}
};

extern std::ostream &operator<<(std::ostream &os, const MySqlStringEscaper &rhs);

struct MySqlDateFormatter {
	const boost::uint64_t &time;

	explicit MySqlDateFormatter(const boost::uint64_t &time_)
		: time(time_)
	{
	}
};

extern std::ostream &operator<<(std::ostream &os, const MySqlDateFormatter &rhs);

}

#endif
