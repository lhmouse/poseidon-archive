// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2016, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_CRC32_HPP_
#define POSEIDON_CRC32_HPP_

#include <string>
#include <cstring>
#include <cstddef>
#include <boost/cstdint.hpp>
#include <ostream>

namespace Poseidon {

typedef boost::uint32_t Crc32;

class Crc32_streambuf : public std::streambuf {
private:
	boost::uint32_t m_reg;

private:
	Crc32_streambuf(const Crc32_streambuf &);
	Crc32_streambuf &operator=(const Crc32_streambuf &);

public:
	Crc32_streambuf();
	~Crc32_streambuf() OVERRIDE;

protected:
	int_type overflow(int_type c = traits_type::eof()) OVERRIDE;

public:
	void clear() NOEXCEPT;
	Crc32 finalize();
};

class Crc32_ostream : public std::ostream {
private:
	Crc32_streambuf m_sb;

public:
	Crc32_ostream()
		: std::ostream(&m_sb)
	{
	}
	~Crc32_ostream() OVERRIDE;

public:
	Crc32_streambuf *rdbuf() const {
		return const_cast<Crc32_streambuf *>(&m_sb);
	}

	void clear() NOEXCEPT {
		return rdbuf()->clear();
	}
	Crc32 finalize(){
		return rdbuf()->finalize();
	}
};

}

#endif
