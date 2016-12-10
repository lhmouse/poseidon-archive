// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2016, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_SHA256_HPP_
#define POSEIDON_SHA256_HPP_

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

public:
	Sha256_streambuf();
	~Sha256_streambuf() OVERRIDE;

protected:
	int_type overflow(int_type c = traits_type::eof()) OVERRIDE;

public:
	Sha256 finalize();

	void swap(Sha256_streambuf &rhs) NOEXCEPT {
		using std::swap;
		swap(m_reg, rhs.m_reg);
	}
};

inline void swap(Sha256_streambuf &lhs, Sha256_streambuf &rhs) NOEXCEPT {
	lhs.swap(rhs);
}

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

	Sha256 finalize(){
		return rdbuf()->finalize();
	}

#ifdef POSEIDON_CXX14
	void swap(Sha256_ostream &rhs) noexcept {
		std::ostream::swap(rhs);
		using std::swap;
		swap(m_sb, rhs.m_sb);
	}
#endif
};

#ifdef POSEIDON_CXX14
inline void swap(Sha256_ostream &lhs, Sha256_ostream &rhs) noexcept {
	lhs.swap(rhs);
}
#endif

inline Sha256 sha256_hash(const void *data, std::size_t size){
	Sha256_ostream s;
	s.exceptions(std::ios::badbit | std::ios::failbit);
	s.write(static_cast<const char *>(data), static_cast<std::streamsize>(size));
	return s.finalize();
}
inline Sha256 sha256_hash(const char *str){
	return sha256_hash(str, std::strlen(str));
}
inline Sha256 sha256_hash(const std::string &str){
	return sha256_hash(str.data(), str.size());
}

}

#endif
