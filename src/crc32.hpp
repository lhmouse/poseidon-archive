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

public:
	Crc32_streambuf();
	~Crc32_streambuf() OVERRIDE;

protected:
	int_type overflow(int_type c = traits_type::eof()) OVERRIDE;

public:
	Crc32 finalize();

	void swap(Crc32_streambuf &rhs) NOEXCEPT {
		using std::swap;
		swap(m_reg, rhs.m_reg);
	}
};

inline void swap(Crc32_streambuf &lhs, Crc32_streambuf &rhs) NOEXCEPT {
	lhs.swap(rhs);
}

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

	Crc32 finalize(){
		return rdbuf()->finalize();
	}

#ifdef POSEIDON_CXX14
	void swap(Crc32_ostream &rhs) noexcept {
		std::ostream::swap(rhs);
		using std::swap;
		swap(m_sb, rhs.m_sb);
	}
#endif
};

#ifdef POSEIDON_CXX14
inline void swap(Crc32_ostream &lhs, Crc32_ostream &rhs) noexcept {
	lhs.swap(rhs);
}
#endif

inline Crc32 crc32_hash(const void *data, std::size_t size){
	Crc32_ostream s;
	s.exceptions(std::ios::badbit | std::ios::failbit);
	s.write(static_cast<const char *>(data), static_cast<std::streamsize>(size));
	return s.finalize();
}
inline Crc32 crc32_hash(const char *str){
	return crc32_hash(str, std::strlen(str));
}
inline Crc32 crc32_hash(const std::string &str){
	return crc32_hash(str.data(), str.size());
}

}

#endif
