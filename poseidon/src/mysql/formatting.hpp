// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_MYSQL_FORMATTING_HPP_
#define POSEIDON_MYSQL_FORMATTING_HPP_

#include "../uuid.hpp"
#include <iosfwd>
#include <string>
#include <boost/cstdint.hpp>

namespace Poseidon {
namespace Mysql {

class String_escaper {
private:
	const std::string &m_ref;

public:
	explicit CONSTEXPR String_escaper(const std::string &ref)
		: m_ref(ref)
	{
		//
	}

public:
	const std::string & get() const {
		return m_ref;
	}
};

extern std::ostream & operator<<(std::ostream &os, const String_escaper &rhs);

class Date_time_formatter {
private:
	const std::uint64_t &m_ref;

public:
	explicit CONSTEXPR Date_time_formatter(const std::uint64_t &ref)
		: m_ref(ref)
	{
		//
	}

public:
	const std::uint64_t & get() const {
		return m_ref;
	}
};

extern std::ostream & operator<<(std::ostream &os, const Date_time_formatter &rhs);

class Uuid_formatter {
private:
	const Uuid &m_ref;

public:
	explicit CONSTEXPR Uuid_formatter(const Uuid &ref)
		: m_ref(ref)
	{
		//
	}

public:
	const Uuid & get() const {
		return m_ref;
	}
};

extern std::ostream & operator<<(std::ostream &os, const Uuid_formatter &rhs);

}
}

#endif
