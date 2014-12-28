// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_SHARED_NTS_HPP_
#define POSEIDON_SHARED_NTS_HPP_

#include "cxx_ver.hpp"
#include <string>
#include <iosfwd>
#include <cstddef>
#include <cstring>
#include <boost/shared_ptr.hpp>

namespace Poseidon {

class SharedNts {
public:
	static SharedNts observe(const char *str);

private:
	boost::shared_ptr<const char> m_ptr;

public:
	SharedNts() NOEXCEPT {
		assign("");
	}
	SharedNts(const char *str, std::size_t len){
		assign(str, len);
	}
	explicit SharedNts(const char *str){
		assign(str);
	}
	explicit SharedNts(const std::string &str){
		assign(str);
	}

public:
	void assign(const char *str, std::size_t len);
	void assign(const char *str){
		assign(str, std::strlen(str));
	}
	void assign(const std::string &str){
		assign(str.data(), str.size());
	}

	bool empty() const {
		return m_ptr.get()[0] == 0;
	}
	const char *get() const {
		return m_ptr.get();
	}

	void swap(SharedNts &rhs) NOEXCEPT {
		m_ptr.swap(rhs.m_ptr);
	}

public:
	operator const char *() const {
		return get();
	}
};

inline bool operator==(const SharedNts &lhs, const SharedNts &rhs){
	return std::strcmp(lhs.get(), rhs.get()) == 0;
}
inline bool operator==(const SharedNts &lhs, const char *rhs){
	return std::strcmp(lhs.get(), rhs) == 0;
}
inline bool operator==(const SharedNts &lhs, const std::string &rhs){
	return lhs.get() == rhs;
}
inline bool operator==(const char *lhs, const SharedNts &rhs){
	return std::strcmp(lhs, rhs.get()) == 0;
}
inline bool operator==(const std::string &lhs, const SharedNts &rhs){
	return lhs == rhs.get();
}

inline bool operator!=(const SharedNts &lhs, const SharedNts &rhs){
	return std::strcmp(lhs.get(), rhs.get()) != 0;
}
inline bool operator!=(const SharedNts &lhs, const char *rhs){
	return std::strcmp(lhs.get(), rhs) != 0;
}
inline bool operator!=(const SharedNts &lhs, const std::string &rhs){
	return lhs.get() != rhs;
}
inline bool operator!=(const char *lhs, const SharedNts &rhs){
	return std::strcmp(lhs, rhs.get()) != 0;
}
inline bool operator!=(const std::string &lhs, const SharedNts &rhs){
	return lhs != rhs.get();
}

inline bool operator<(const SharedNts &lhs, const SharedNts &rhs){
	return std::strcmp(lhs.get(), rhs.get()) < 0;
}
inline bool operator<(const SharedNts &lhs, const char *rhs){
	return std::strcmp(lhs.get(), rhs) < 0;
}
inline bool operator<(const SharedNts &lhs, const std::string &rhs){
	return lhs.get() < rhs;
}
inline bool operator<(const char *lhs, const SharedNts &rhs){
	return std::strcmp(lhs, rhs.get()) < 0;
}
inline bool operator<(const std::string &lhs, const SharedNts &rhs){
	return lhs < rhs.get();
}

inline bool operator>(const SharedNts &lhs, const SharedNts &rhs){
	return std::strcmp(lhs.get(), rhs.get()) > 0;
}
inline bool operator>(const SharedNts &lhs, const char *rhs){
	return std::strcmp(lhs.get(), rhs) > 0;
}
inline bool operator>(const SharedNts &lhs, const std::string &rhs){
	return lhs.get() > rhs;
}
inline bool operator>(const char *lhs, const SharedNts &rhs){
	return std::strcmp(lhs, rhs.get()) > 0;
}
inline bool operator>(const std::string &lhs, const SharedNts &rhs){
	return lhs > rhs.get();
}

inline bool operator<=(const SharedNts &lhs, const SharedNts &rhs){
	return std::strcmp(lhs.get(), rhs.get()) <= 0;
}
inline bool operator<=(const SharedNts &lhs, const char *rhs){
	return std::strcmp(lhs.get(), rhs) <= 0;
}
inline bool operator<=(const SharedNts &lhs, const std::string &rhs){
	return lhs.get() <= rhs;
}
inline bool operator<=(const char *lhs, const SharedNts &rhs){
	return std::strcmp(lhs, rhs.get()) <= 0;
}
inline bool operator<=(const std::string &lhs, const SharedNts &rhs){
	return lhs <= rhs.get();
}

inline bool operator>=(const SharedNts &lhs, const SharedNts &rhs){
	return std::strcmp(lhs.get(), rhs.get()) >= 0;
}
inline bool operator>=(const SharedNts &lhs, const char *rhs){
	return std::strcmp(lhs.get(), rhs) >= 0;
}
inline bool operator>=(const SharedNts &lhs, const std::string &rhs){
	return lhs.get() >= rhs;
}
inline bool operator>=(const char *lhs, const SharedNts &rhs){
	return std::strcmp(lhs, rhs.get()) >= 0;
}
inline bool operator>=(const std::string &lhs, const SharedNts &rhs){
	return lhs >= rhs.get();
}

inline void swap(SharedNts &lhs, SharedNts &rhs) NOEXCEPT {
	lhs.swap(rhs);
}

extern std::ostream &operator<<(std::ostream &os, const SharedNts &rhs);
extern std::wostream &operator<<(std::wostream &os, const SharedNts &rhs);

}

#endif
