// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

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
	static const Uuid & min() NOEXCEPT;
	static const Uuid & max() NOEXCEPT;

	static Uuid random() NOEXCEPT;

private:
	std::array<unsigned char, 16> m_bytes;

public:
	CONSTEXPR Uuid() NOEXCEPT
		: m_bytes()
	{
		//
	}
	explicit Uuid(const unsigned char (&bytes)[16]){
		std::memcpy(m_bytes.data(), bytes, 16);
	}
	explicit Uuid(const std::array<unsigned char, 16> &bytes){
		std::memcpy(m_bytes.data(), bytes.data(), 16);
	}
	// 字符串不合法则抛出异常。
	explicit Uuid(const char (&str)[36]);
	explicit Uuid(const std::string &str);

public:
	bool is_null() const {
		return m_bytes == min().m_bytes;
	}

	const std::array<unsigned char, 16> & as_array() const {
		return m_bytes;
	}
	std::array<unsigned char, 16> & as_array(){
		return m_bytes;
	}

	const unsigned char * begin() const {
		return m_bytes.data();
	}
	unsigned char * begin(){
		return m_bytes.data();
	}
	const unsigned char * end() const {
		return m_bytes.data() + m_bytes.size();
	}
	unsigned char * end(){
		return m_bytes.data() + m_bytes.size();
	}
	const unsigned char * data() const {
		return m_bytes.data();
	}
	unsigned char * data(){
		return m_bytes.data();
	}
	std::size_t size() const {
		return m_bytes.size();
	}

	void to_string(char (&str)[36], bool upper_case = true) const;
	void to_string(std::string &str, bool upper_case = true) const;
	std::string to_string(bool upper_case = true) const;
	bool from_string(const char (&str)[36]);
	bool from_string(const std::string &str);

public:
#ifdef POSEIDON_CXX11
	explicit operator bool() const noexcept {
		return !is_null();
	}
#else
	typedef bool (Uuid::*Dummy_bool_)() const;
	operator Dummy_bool_() const NOEXCEPT {
		return !is_null() ? &Uuid::is_null : 0;
	}
#endif

	operator const std::array<unsigned char, 16> &() const {
		return as_array();
	}
	operator std::array<unsigned char, 16> &(){
		return as_array();
	}

	const unsigned char & operator[](std::size_t index) const {
		return as_array()[index];
	}
	unsigned char & operator[](std::size_t index){
		return as_array()[index];
	}
};

inline bool operator==(const Uuid &lhs, const Uuid &rhs){
	return std::memcmp(lhs.data(), rhs.data(), lhs.size()) == 0;
}
inline bool operator!=(const Uuid &lhs, const Uuid &rhs){
	return std::memcmp(lhs.data(), rhs.data(), lhs.size()) != 0;
}
inline bool operator<(const Uuid &lhs, const Uuid &rhs){
	return std::memcmp(lhs.data(), rhs.data(), lhs.size()) < 0;
}
inline bool operator>(const Uuid &lhs, const Uuid &rhs){
	return std::memcmp(lhs.data(), rhs.data(), lhs.size()) > 0;
}
inline bool operator<=(const Uuid &lhs, const Uuid &rhs){
	return std::memcmp(lhs.data(), rhs.data(), lhs.size()) <= 0;
}
inline bool operator>=(const Uuid &lhs, const Uuid &rhs){
	return std::memcmp(lhs.data(), rhs.data(), lhs.size()) >= 0;
}

extern std::ostream & operator<<(std::ostream &os, const Uuid &rhs);

}

#endif
