// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_HEX_PRINTER_HPP_
#define POSEIDON_HEX_PRINTER_HPP_

#include "cxx_ver.hpp"
#include <iosfwd>
#include <cstddef>

namespace Poseidon {

class Hex_printer {
private:
	const void *m_data;
	std::size_t m_size;
	char m_delimiter;

public:
	CONSTEXPR Hex_printer(const void *data, std::size_t size, char delimiter = ' ')
		: m_data(data), m_size(size), m_delimiter(delimiter)
	{
		//
	}

public:
	const void * get_data() const {
		return m_data;
	}
	std::size_t get_size() const {
		return m_size;
	}
	char get_delimiter() const {
		return m_delimiter;
	}
};

extern std::ostream & operator<<(std::ostream &os, const Hex_printer &rhs);

}

#endif
