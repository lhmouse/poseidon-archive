// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2017, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_UUID_HPP_
#define POSEIDON_UUID_HPP_

#include "cxx_ver.hpp"
#include <iosfwd>
#include <string>
#include <cstring>
#include <boost/array.hpp>
#include <boost/cstdint.hpp>

namespace Poseidon {

class Uuid {
public:
	static const Uuid &min() NOEXCEPT;
	static const Uuid &max() NOEXCEPT;

	static Uuid random() NOEXCEPT;

private:
	unsigned char m_bytes[16];

public:
	CONSTEXPR Uuid() NOEXCEPT
		: m_bytes()
	{ }
	explicit Uuid(const unsigned char (&bytes)[16]){
		std::memcpy(m_bytes, bytes, 16);
	}
	explicit Uuid(const boost::array<unsigned char, 16> &bytes){
		std::memcpy(m_bytes, bytes.data(), 16);
	}
	// 字符串不合法则抛出异常。
	explicit Uuid(const char (&str)[36]);
	explicit Uuid(const std::string &str);

public:
	CONSTEXPR const unsigned char *begin() const {
		return m_bytes;
	}
	unsigned char *begin(){
		return m_bytes;
	}
	CONSTEXPR const unsigned char *end() const {
		return m_bytes + 16;
	}
	unsigned char *end(){
		return m_bytes + 16;
	}
	CONSTEXPR const unsigned char *data() const {
		return m_bytes;
	}
	unsigned char *data(){
		return m_bytes;
	}
	CONSTEXPR std::size_t size() const {
		return 16;
	}

	void to_string(char (&str)[36], bool upper_case = true) const;
	void to_string(std::string &str, bool upper_case = true) const;
	std::string to_string(bool upper_case = true) const;
	bool from_string(const char (&str)[36]);
	bool from_string(const std::string &str);

public:
	const unsigned char &operator[](std::size_t index) const {
		return m_bytes[index];
	}
	unsigned char &operator[](std::size_t index){
		return m_bytes[index];
	}
};

inline bool operator==(const Uuid &lhs, const Uuid &rhs){
	return std::memcmp(lhs.begin(), rhs.begin(), 16) == 0;
}
inline bool operator!=(const Uuid &lhs, const Uuid &rhs){
	return std::memcmp(lhs.begin(), rhs.begin(), 16) != 0;
}
inline bool operator<(const Uuid &lhs, const Uuid &rhs){
	return std::memcmp(lhs.begin(), rhs.begin(), 16) < 0;
}
inline bool operator>(const Uuid &lhs, const Uuid &rhs){
	return std::memcmp(lhs.begin(), rhs.begin(), 16) > 0;
}
inline bool operator<=(const Uuid &lhs, const Uuid &rhs){
	return std::memcmp(lhs.begin(), rhs.begin(), 16) <= 0;
}
inline bool operator>=(const Uuid &lhs, const Uuid &rhs){
	return std::memcmp(lhs.begin(), rhs.begin(), 16) >= 0;
}

extern std::ostream &operator<<(std::ostream &os, const Uuid &rhs);

}

#endif
