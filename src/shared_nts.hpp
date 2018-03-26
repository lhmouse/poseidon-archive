// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_SHARED_NTS_HPP_
#define POSEIDON_SHARED_NTS_HPP_

#include "cxx_ver.hpp"
#include <string>
#include <iosfwd>
#include <cstddef>
#include <cstring>
#include <boost/shared_ptr.hpp>

namespace Poseidon {

class Shared_nts {
public:
	static Shared_nts view(const char *str){
		return Shared_nts(boost::shared_ptr<void>(), str);
	}

private:
	boost::shared_ptr<const char> m_ptr;

public:
	Shared_nts() NOEXCEPT {
		assign("", 0);
	}
	Shared_nts(const char *str, std::size_t len){
		assign(str, len);
	}
	explicit Shared_nts(const char *str){
		assign(str);
	}
	explicit Shared_nts(const std::string &str){
		assign(str);
	}
	template<typename T>
	Shared_nts(boost::shared_ptr<T> sp, const char *str){
		assign(STD_MOVE_IDN(sp), str);
	}

	Shared_nts(const Shared_nts &rhs) NOEXCEPT
		: m_ptr(rhs.m_ptr)
	{
		//
	}
	Shared_nts &operator=(const Shared_nts &rhs) NOEXCEPT {
		m_ptr = rhs.m_ptr;
		return *this;
	}
#ifdef POSEIDON_CXX11
	Shared_nts(Shared_nts &&rhs) NOEXCEPT {
		assign("", 0);
		swap(rhs);
	}
	Shared_nts &operator=(Shared_nts &&rhs) NOEXCEPT {
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
	const char *get() const {
		return m_ptr.get();
	}

	void swap(Shared_nts &rhs) NOEXCEPT {
		using std::swap;
		swap(m_ptr, rhs.m_ptr);
	}

public:
	operator const char *() const {
		return get();
	}
};

inline bool operator==(const Shared_nts &lhs, const Shared_nts &rhs){
	return std::strcmp(lhs.get(), rhs.get()) == 0;
}
inline bool operator==(const Shared_nts &lhs, const char *rhs){
	return std::strcmp(lhs.get(), rhs) == 0;
}
inline bool operator==(const Shared_nts &lhs, const std::string &rhs){
	return lhs.get() == rhs;
}
inline bool operator==(const char *lhs, const Shared_nts &rhs){
	return std::strcmp(lhs, rhs.get()) == 0;
}
inline bool operator==(const std::string &lhs, const Shared_nts &rhs){
	return lhs == rhs.get();
}

inline bool operator!=(const Shared_nts &lhs, const Shared_nts &rhs){
	return std::strcmp(lhs.get(), rhs.get()) != 0;
}
inline bool operator!=(const Shared_nts &lhs, const char *rhs){
	return std::strcmp(lhs.get(), rhs) != 0;
}
inline bool operator!=(const Shared_nts &lhs, const std::string &rhs){
	return lhs.get() != rhs;
}
inline bool operator!=(const char *lhs, const Shared_nts &rhs){
	return std::strcmp(lhs, rhs.get()) != 0;
}
inline bool operator!=(const std::string &lhs, const Shared_nts &rhs){
	return lhs != rhs.get();
}

inline bool operator<(const Shared_nts &lhs, const Shared_nts &rhs){
	return std::strcmp(lhs.get(), rhs.get()) < 0;
}
inline bool operator<(const Shared_nts &lhs, const char *rhs){
	return std::strcmp(lhs.get(), rhs) < 0;
}
inline bool operator<(const Shared_nts &lhs, const std::string &rhs){
	return lhs.get() < rhs;
}
inline bool operator<(const char *lhs, const Shared_nts &rhs){
	return std::strcmp(lhs, rhs.get()) < 0;
}
inline bool operator<(const std::string &lhs, const Shared_nts &rhs){
	return lhs < rhs.get();
}

inline bool operator>(const Shared_nts &lhs, const Shared_nts &rhs){
	return std::strcmp(lhs.get(), rhs.get()) > 0;
}
inline bool operator>(const Shared_nts &lhs, const char *rhs){
	return std::strcmp(lhs.get(), rhs) > 0;
}
inline bool operator>(const Shared_nts &lhs, const std::string &rhs){
	return lhs.get() > rhs;
}
inline bool operator>(const char *lhs, const Shared_nts &rhs){
	return std::strcmp(lhs, rhs.get()) > 0;
}
inline bool operator>(const std::string &lhs, const Shared_nts &rhs){
	return lhs > rhs.get();
}

inline bool operator<=(const Shared_nts &lhs, const Shared_nts &rhs){
	return std::strcmp(lhs.get(), rhs.get()) <= 0;
}
inline bool operator<=(const Shared_nts &lhs, const char *rhs){
	return std::strcmp(lhs.get(), rhs) <= 0;
}
inline bool operator<=(const Shared_nts &lhs, const std::string &rhs){
	return lhs.get() <= rhs;
}
inline bool operator<=(const char *lhs, const Shared_nts &rhs){
	return std::strcmp(lhs, rhs.get()) <= 0;
}
inline bool operator<=(const std::string &lhs, const Shared_nts &rhs){
	return lhs <= rhs.get();
}

inline bool operator>=(const Shared_nts &lhs, const Shared_nts &rhs){
	return std::strcmp(lhs.get(), rhs.get()) >= 0;
}
inline bool operator>=(const Shared_nts &lhs, const char *rhs){
	return std::strcmp(lhs.get(), rhs) >= 0;
}
inline bool operator>=(const Shared_nts &lhs, const std::string &rhs){
	return lhs.get() >= rhs;
}
inline bool operator>=(const char *lhs, const Shared_nts &rhs){
	return std::strcmp(lhs, rhs.get()) >= 0;
}
inline bool operator>=(const std::string &lhs, const Shared_nts &rhs){
	return lhs >= rhs.get();
}

inline void swap(Shared_nts &lhs, Shared_nts &rhs) NOEXCEPT {
	lhs.swap(rhs);
}

extern std::istream &operator>>(std::istream &is, Shared_nts &rhs);
extern std::ostream &operator<<(std::ostream &os, const Shared_nts &rhs);

// Shared String LITeral
template<std::size_t N>
inline Shared_nts sslit(const char (&str)[N]){
	return Shared_nts::view(str);
}

}

#endif
