// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2016, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_MD5_HPP_
#define POSEIDON_MD5_HPP_

#include <string>
#include <cstring>
#include <cstddef>
#include <boost/cstdint.hpp>
#include <boost/array.hpp>
#include <ostream>

namespace Poseidon {

typedef boost::array<boost::uint8_t, 16> Md5;

class Md5_streambuf : public std::streambuf {
private:
	boost::array<boost::uint32_t, 4> m_reg;
	boost::uint64_t m_bytes;
	boost::array<char, 64> m_chunk;

public:
	Md5_streambuf();
	~Md5_streambuf() OVERRIDE;

protected:
	int_type overflow(int_type c = traits_type::eof()) OVERRIDE;

public:
	Md5 finalize();

	void swap(Md5_streambuf &rhs) NOEXCEPT {
		using std::swap;
		swap(m_reg, rhs.m_reg);
	}
};

inline void swap(Md5_streambuf &lhs, Md5_streambuf &rhs) NOEXCEPT {
	lhs.swap(rhs);
}

class Md5_ostream : public std::ostream {
private:
	Md5_streambuf m_sb;

public:
	Md5_ostream()
		: std::ostream(&m_sb)
	{
	}
	~Md5_ostream() OVERRIDE;

public:
	Md5_streambuf *rdbuf() const {
		return const_cast<Md5_streambuf *>(&m_sb);
	}

	Md5 finalize(){
		return rdbuf()->finalize();
	}

#ifdef POSEIDON_CXX14
	void swap(Md5_ostream &rhs) noexcept {
		std::ostream::swap(rhs);
		using std::swap;
		swap(m_sb, rhs.m_sb);
	}
#endif
};

#ifdef POSEIDON_CXX14
inline void swap(Md5_ostream &lhs, Md5_ostream &rhs) noexcept {
	lhs.swap(rhs);
}
#endif

inline Md5 md5_hash(const void *data, std::size_t size){
	Md5_ostream s;
	s.exceptions(std::ios::badbit | std::ios::failbit);
	s.write(static_cast<const char *>(data), static_cast<std::streamsize>(size));
	return s.finalize();
}
inline Md5 md5_hash(const char *str){
	return md5_hash(str, std::strlen(str));
}
inline Md5 md5_hash(const std::string &str){
	return md5_hash(str.data(), str.size());
}

}

#endif
