// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_UUID_HPP_
#define POSEIDON_UUID_HPP_

#include <string>
#include <iosfwd>
#include <cstring>

namespace Poseidon {

class Uuid {
public:
	static Uuid generate();

private:
	unsigned char m_bytes[16];

public:
	Uuid(){
		std::memset(m_bytes, 0, 16);
	}
	explicit Uuid(const unsigned char (&bytes)[16]){
		std::memcpy(m_bytes, bytes, 16);
	}
	explicit Uuid(const std::string &str){
		fromString(str);
	}

public:
	std::string toString() const;
	bool fromString(const std::string &str);

	const unsigned char (&getBytes() const)[16] {
		return m_bytes;
	}
	unsigned char (&getBytes())[16] {
		return m_bytes;
	}

public:
	const unsigned char &operator[](unsigned index) const {
		return m_bytes[index];
	}
	unsigned char &operator[](unsigned index){
		return m_bytes[index];
	}
};

inline bool operator==(const Uuid &lhs, const Uuid &rhs){
	return std::memcmp(lhs.getBytes(), rhs.getBytes(), 16) == 0;
}
inline bool operator!=(const Uuid &lhs, const Uuid &rhs){
	return std::memcmp(lhs.getBytes(), rhs.getBytes(), 16) != 0;
}
inline bool operator<(const Uuid &lhs, const Uuid &rhs){
	return std::memcmp(lhs.getBytes(), rhs.getBytes(), 16) < 0;
}
inline bool operator>(const Uuid &lhs, const Uuid &rhs){
	return std::memcmp(lhs.getBytes(), rhs.getBytes(), 16) > 0;
}
inline bool operator<=(const Uuid &lhs, const Uuid &rhs){
	return std::memcmp(lhs.getBytes(), rhs.getBytes(), 16) <= 0;
}
inline bool operator>=(const Uuid &lhs, const Uuid &rhs){
	return std::memcmp(lhs.getBytes(), rhs.getBytes(), 16) >= 0;
}

extern std::ostream &operator<<(std::ostream &os, const Uuid &rhs);

}

#endif
