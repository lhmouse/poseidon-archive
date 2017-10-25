// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2017, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_MYSQL_FORMATTING_HPP_
#define POSEIDON_MYSQL_FORMATTING_HPP_

#include "../fwd.hpp"
#include <iosfwd>
#include <string>
#include <boost/cstdint.hpp>

namespace Poseidon {
namespace MySql {

class StringEscaper {
private:
	const std::string &m_ref;

public:
	explicit CONSTEXPR StringEscaper(const std::string &ref)
		: m_ref(ref)
	{ }

public:
	const std::string &get() const {
		return m_ref;
	}
};

extern std::ostream &operator<<(std::ostream &os, const StringEscaper &rhs);

class DateTimeFormatter {
private:
	const boost::uint64_t &m_ref;

public:
	explicit CONSTEXPR DateTimeFormatter(const boost::uint64_t &ref)
		: m_ref(ref)
	{ }

public:
	const boost::uint64_t &get() const {
		return m_ref;
	}
};

extern std::ostream &operator<<(std::ostream &os, const DateTimeFormatter &rhs);

class UuidFormatter {
private:
	const Uuid &m_ref;

public:
	explicit CONSTEXPR UuidFormatter(const Uuid &ref)
		: m_ref(ref)
	{ }

public:
	const Uuid &get() const {
		return m_ref;
	}
};

extern std::ostream &operator<<(std::ostream &os, const UuidFormatter &rhs);

}
}

#endif
