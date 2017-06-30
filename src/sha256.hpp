// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2017, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_SHA256_HPP_
#define POSEIDON_SHA256_HPP_

#include "cxx_ver.hpp"
#include <string>
#include <cstring>
#include <cstddef>
#include <boost/cstdint.hpp>
#include <boost/array.hpp>
#include <ostream>

namespace Poseidon {

typedef boost::array<boost::uint8_t, 32> Sha256;

class Sha256_streambuf : public std::streambuf {
private:
	boost::array<boost::uint32_t, 8> m_reg;
	boost::uint64_t m_bytes;
	boost::array<char, 64> m_chunk;

private:
	Sha256_streambuf(const Sha256_streambuf &);
	Sha256_streambuf &operator=(const Sha256_streambuf &);

public:
	Sha256_streambuf();
	~Sha256_streambuf() OVERRIDE;

private:
	void eat_chunk();

protected:
	int_type overflow(int_type c = traits_type::eof()) OVERRIDE;

public:
	void reset() NOEXCEPT;
	Sha256 finalize();
};

class Sha256_ostream : public std::ostream {
private:
	Sha256_streambuf m_sb;

public:
	Sha256_ostream()
		: std::ostream(&m_sb)
	{
	}
	~Sha256_ostream() OVERRIDE;

public:
	Sha256_streambuf *rdbuf() const {
		return const_cast<Sha256_streambuf *>(&m_sb);
	}

	void reset() NOEXCEPT {
		return rdbuf()->reset();
	}
	Sha256 finalize(){
		return rdbuf()->finalize();
	}
};

}

#endif
