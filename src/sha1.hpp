// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2016, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_SHA1_HPP_
#define POSEIDON_SHA1_HPP_

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

public:
	Sha1_streambuf();
	~Sha1_streambuf() OVERRIDE;

protected:
	int_type overflow(int_type c = traits_type::eof()) OVERRIDE;

public:
	Sha1 finalize();

	void swap(Sha1_streambuf &rhs) NOEXCEPT {
		using std::swap;
		swap(m_reg, rhs.m_reg);
	}
};

inline void swap(Sha1_streambuf &lhs, Sha1_streambuf &rhs) NOEXCEPT {
	lhs.swap(rhs);
}

class Sha1_ostream : public std::ostream {
private:
	Sha1_streambuf m_sb;

public:
	Sha1_ostream()
		: std::ostream(&m_sb)
	{
	}
	~Sha1_ostream() OVERRIDE;

public:
	Sha1_streambuf *rdbuf() const {
		return const_cast<Sha1_streambuf *>(&m_sb);
	}

	Sha1 finalize(){
		return rdbuf()->finalize();
	}

#ifdef POSEIDON_CXX14
	void swap(Sha1_ostream &rhs) noexcept {
		std::ostream::swap(rhs);
		using std::swap;
		swap(m_sb, rhs.m_sb);
	}
#endif
};

#ifdef POSEIDON_CXX14
inline void swap(Sha1_ostream &lhs, Sha1_ostream &rhs) noexcept {
	lhs.swap(rhs);
}
#endif

inline Sha1 sha1_hash(const void *data, std::size_t size){
	Sha1_ostream s;
	s.exceptions(std::ios::badbit | std::ios::failbit);
	s.write(static_cast<const char *>(data), static_cast<std::streamsize>(size));
	return s.finalize();
}
inline Sha1 sha1_hash(const char *str){
	return sha1_hash(str, std::strlen(str));
}
inline Sha1 sha1_hash(const std::string &str){
	return sha1_hash(str.data(), str.size());
}

}

#endif
