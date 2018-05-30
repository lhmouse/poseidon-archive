// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_MD5_HPP_
#define POSEIDON_MD5_HPP_

#include "cxx_ver.hpp"
#include <string>
#include <cstring>
#include <cstddef>
#include <boost/cstdint.hpp>
#include <boost/array.hpp>
#include <ostream>

namespace Poseidon {

typedef std::array<std::uint8_t, 16> Md5;

class Md5_streambuf : public std::streambuf {
private:
	std::array<std::uint32_t, 4> m_reg;
	std::uint64_t m_bytes;
	std::array<char, 64> m_chunk;

private:
	Md5_streambuf(const Md5_streambuf &);
	Md5_streambuf & operator=(const Md5_streambuf &);

public:
	Md5_streambuf();
	~Md5_streambuf() OVERRIDE;

private:
	void eat_chunk();

protected:
	int_type overflow(int_type c = traits_type::eof()) OVERRIDE;

public:
	void reset() NOEXCEPT;
	Md5 finalize();
};

class Md5_ostream : public std::ostream {
private:
	Md5_streambuf m_sb;

public:
	Md5_ostream()
		: std::ostream(&m_sb)
	{
		//
	}
	~Md5_ostream() OVERRIDE;

public:
	Md5_streambuf * rdbuf() const {
		return const_cast<Md5_streambuf *>(&m_sb);
	}

	void reset() NOEXCEPT {
		return rdbuf()->reset();
	}
	Md5 finalize(){
		return rdbuf()->finalize();
	}
};

}

#endif
