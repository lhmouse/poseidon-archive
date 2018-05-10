// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_SHA1_HPP_
#define POSEIDON_SHA1_HPP_

#include "cxx_ver.hpp"
#include <string>
#include <cstring>
#include <cstddef>
#include <boost/cstdint.hpp>
#include <boost/array.hpp>
#include <ostream>

namespace Poseidon {

typedef boost::array<boost::uint8_t, 20> Sha1;

class Sha1_streambuf : public std::streambuf {
private:
	boost::array<boost::uint32_t, 5> m_reg;
	boost::uint64_t m_bytes;
	boost::array<char, 64> m_chunk;

private:
	Sha1_streambuf(const Sha1_streambuf &);
	Sha1_streambuf &operator=(const Sha1_streambuf &);

public:
	Sha1_streambuf();
	~Sha1_streambuf() OVERRIDE;

private:
	void eat_chunk();

protected:
	int_type overflow(int_type c = traits_type::eof()) OVERRIDE;

public:
	void reset() NOEXCEPT;
	Sha1 finalize();
};

class Sha1_ostream : public std::ostream {
private:
	Sha1_streambuf m_sb;

public:
	Sha1_ostream()
		: std::ostream(&m_sb)
	{
		//
	}
	~Sha1_ostream() OVERRIDE;

public:
	Sha1_streambuf *rdbuf() const {
		return const_cast<Sha1_streambuf *>(&m_sb);
	}

	void reset() NOEXCEPT {
		return rdbuf()->reset();
	}
	Sha1 finalize(){
		return rdbuf()->finalize();
	}
};

}

#endif
