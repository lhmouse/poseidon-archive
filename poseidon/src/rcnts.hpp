// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_RCNTS_HPP_
#define POSEIDON_RCNTS_HPP_

#include "cxx_ver.hpp"
#include <string>
#include <iosfwd>
#include <cstddef>
#include <cstring>
#include <boost/shared_ptr.hpp>

namespace Poseidon {

class Rcnts {
public:
	static Rcnts view(const char *str){
		return Rcnts(boost::shared_ptr<void>(), str);
	}

private:
	boost::shared_ptr<const char> m_ptr;

public:
	Rcnts() NOEXCEPT {
		assign("", 0);
	}
	Rcnts(const char *str, std::size_t len){
		assign(str, len);
	}
	explicit Rcnts(const char *str){
		assign(str);
	}
	explicit Rcnts(const std::string &str){
		assign(str);
	}
	template<typename T>
	Rcnts(boost::shared_ptr<T> sp, const char *str){
		assign(STD_MOVE_IDN(sp), str);
	}

	Rcnts(const Rcnts &rhs) NOEXCEPT
		: m_ptr(rhs.m_ptr)
	{
		//
	}
	Rcnts & operator=(const Rcnts &rhs) NOEXCEPT {
		m_ptr = rhs.m_ptr;
		return *this;
	}
#ifdef POSEIDON_CXX11
	Rcnts(Rcnts &&rhs) NOEXCEPT {
		assign("", 0);
		swap(rhs);
	}
	Rcnts & operator=(Rcnts &&rhs) NOEXCEPT {
		swap(rhs);
		return *this;
	}
#endif

public:
	void assign(const char *str, std::size_t len);
	void assign(const char *str){
		assign(str, std::strlen(str));
	}
	void assign(const std::string &str){
		assign(str.data(), str.size());
	}
	template<typename T>
	void assign(boost::shared_ptr<T> sp, const char *str){
		m_ptr.reset(STD_MOVE_IDN(sp), str ? str : "");
	}

	bool empty() const {
		return m_ptr.get()[0] == 0;
	}
	const char * get() const {
		return m_ptr.get();
	}

	void swap(Rcnts &rhs) NOEXCEPT {
		using std::swap;
		swap(m_ptr, rhs.m_ptr);
	}

public:
	operator const char *() const {
		return get();
	}
};

inline bool operator==(const Rcnts &lhs, const Rcnts &rhs){
	return std::strcmp(lhs.get(), rhs.get()) == 0;
}
inline bool operator==(const Rcnts &lhs, const char *rhs){
	return std::strcmp(lhs.get(), rhs) == 0;
}
inline bool operator==(const Rcnts &lhs, const std::string &rhs){
	return lhs.get() == rhs;
}
inline bool operator==(const char *lhs, const Rcnts &rhs){
	return std::strcmp(lhs, rhs.get()) == 0;
}
inline bool operator==(const std::string &lhs, const Rcnts &rhs){
	return lhs == rhs.get();
}

inline bool operator!=(const Rcnts &lhs, const Rcnts &rhs){
	return std::strcmp(lhs.get(), rhs.get()) != 0;
}
inline bool operator!=(const Rcnts &lhs, const char *rhs){
	return std::strcmp(lhs.get(), rhs) != 0;
}
inline bool operator!=(const Rcnts &lhs, const std::string &rhs){
	return lhs.get() != rhs;
}
inline bool operator!=(const char *lhs, const Rcnts &rhs){
	return std::strcmp(lhs, rhs.get()) != 0;
}
inline bool operator!=(const std::string &lhs, const Rcnts &rhs){
	return lhs != rhs.get();
}

inline bool operator<(const Rcnts &lhs, const Rcnts &rhs){
	return std::strcmp(lhs.get(), rhs.get()) < 0;
}
inline bool operator<(const Rcnts &lhs, const char *rhs){
	return std::strcmp(lhs.get(), rhs) < 0;
}
inline bool operator<(const Rcnts &lhs, const std::string &rhs){
	return lhs.get() < rhs;
}
inline bool operator<(const char *lhs, const Rcnts &rhs){
	return std::strcmp(lhs, rhs.get()) < 0;
}
inline bool operator<(const std::string &lhs, const Rcnts &rhs){
	return lhs < rhs.get();
}

inline bool operator>(const Rcnts &lhs, const Rcnts &rhs){
	return std::strcmp(lhs.get(), rhs.get()) > 0;
}
inline bool operator>(const Rcnts &lhs, const char *rhs){
	return std::strcmp(lhs.get(), rhs) > 0;
}
inline bool operator>(const Rcnts &lhs, const std::string &rhs){
	return lhs.get() > rhs;
}
inline bool operator>(const char *lhs, const Rcnts &rhs){
	return std::strcmp(lhs, rhs.get()) > 0;
}
inline bool operator>(const std::string &lhs, const Rcnts &rhs){
	return lhs > rhs.get();
}

inline bool operator<=(const Rcnts &lhs, const Rcnts &rhs){
	return std::strcmp(lhs.get(), rhs.get()) <= 0;
}
inline bool operator<=(const Rcnts &lhs, const char *rhs){
	return std::strcmp(lhs.get(), rhs) <= 0;
}
inline bool operator<=(const Rcnts &lhs, const std::string &rhs){
	return lhs.get() <= rhs;
}
inline bool operator<=(const char *lhs, const Rcnts &rhs){
	return std::strcmp(lhs, rhs.get()) <= 0;
}
inline bool operator<=(const std::string &lhs, const Rcnts &rhs){
	return lhs <= rhs.get();
}

inline bool operator>=(const Rcnts &lhs, const Rcnts &rhs){
	return std::strcmp(lhs.get(), rhs.get()) >= 0;
}
inline bool operator>=(const Rcnts &lhs, const char *rhs){
	return std::strcmp(lhs.get(), rhs) >= 0;
}
inline bool operator>=(const Rcnts &lhs, const std::string &rhs){
	return lhs.get() >= rhs;
}
inline bool operator>=(const char *lhs, const Rcnts &rhs){
	return std::strcmp(lhs, rhs.get()) >= 0;
}
inline bool operator>=(const std::string &lhs, const Rcnts &rhs){
	return lhs >= rhs.get();
}

inline void swap(Rcnts &lhs, Rcnts &rhs) NOEXCEPT {
	lhs.swap(rhs);
}

extern std::istream & operator>>(std::istream &is, Rcnts &rhs);
extern std::ostream & operator<<(std::ostream &os, const Rcnts &rhs);

}

#endif
