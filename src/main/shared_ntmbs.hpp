// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_SHARED_NTMBS_HPP_
#define POSEIDON_SHARED_NTMBS_HPP_

#include "cxx_ver.hpp"
#include <string>
#include <iosfwd>
#include <cstddef>
#include <cstring>
#include <boost/shared_ptr.hpp>

namespace Poseidon {

class SharedNtmbs {
private:
	boost::shared_ptr<const char> m_ptr;

public:
	SharedNtmbs() NOEXCEPT
		: m_ptr(boost::shared_ptr<void>(), "")
	{
	}
	SharedNtmbs(const char *str, bool owning = false)
		: m_ptr(boost::shared_ptr<void>(), str ? str : "")
	{
		if(owning){
			forkOwning();
		}
	}
	SharedNtmbs(const std::string &str, bool owning = false)
		: m_ptr(boost::shared_ptr<void>(), str.c_str())
	{
		if(owning){
			forkOwning();
		}
	}
	SharedNtmbs(const SharedNtmbs &rhs, bool owning){
		m_ptr = rhs.m_ptr;

		if(owning){
			forkOwning();
		}
	}
	SharedNtmbs(Move<SharedNtmbs> rhs, bool owning){
		m_ptr.swap(rhs.m_ptr);

		if(owning){
			forkOwning();
		}
	}

public:
	const char *get() const {
		return m_ptr.get();
	}
	bool empty() const {
		return get()[0] == 0;
	}

	bool isOwning() const {
		return m_ptr.use_count() > 0;
	}
	void forkOwning();

	void swap(SharedNtmbs &rhs) NOEXCEPT {
		m_ptr.swap(rhs.m_ptr);
	}

public:
	operator const char *() const {
		return get();
	}
};

inline bool operator==(const SharedNtmbs &lhs, const SharedNtmbs &rhs){
	return std::strcmp(lhs.get(), rhs.get()) == 0;
}
inline bool operator==(const SharedNtmbs &lhs, const char *rhs){
	return std::strcmp(lhs.get(), rhs) == 0;
}
inline bool operator==(const SharedNtmbs &lhs, const std::string &rhs){
	return lhs.get() == rhs;
}
inline bool operator==(const char *lhs, const SharedNtmbs &rhs){
	return std::strcmp(lhs, rhs.get()) == 0;
}
inline bool operator==(const std::string &lhs, const SharedNtmbs &rhs){
	return lhs == rhs.get();
}

inline bool operator!=(const SharedNtmbs &lhs, const SharedNtmbs &rhs){
	return std::strcmp(lhs.get(), rhs.get()) != 0;
}
inline bool operator!=(const SharedNtmbs &lhs, const char *rhs){
	return std::strcmp(lhs.get(), rhs) != 0;
}
inline bool operator!=(const SharedNtmbs &lhs, const std::string &rhs){
	return lhs.get() != rhs;
}
inline bool operator!=(const char *lhs, const SharedNtmbs &rhs){
	return std::strcmp(lhs, rhs.get()) != 0;
}
inline bool operator!=(const std::string &lhs, const SharedNtmbs &rhs){
	return lhs != rhs.get();
}

inline bool operator<(const SharedNtmbs &lhs, const SharedNtmbs &rhs){
	return std::strcmp(lhs.get(), rhs.get()) < 0;
}
inline bool operator<(const SharedNtmbs &lhs, const char *rhs){
	return std::strcmp(lhs.get(), rhs) < 0;
}
inline bool operator<(const SharedNtmbs &lhs, const std::string &rhs){
	return lhs.get() < rhs;
}
inline bool operator<(const char *lhs, const SharedNtmbs &rhs){
	return std::strcmp(lhs, rhs.get()) < 0;
}
inline bool operator<(const std::string &lhs, const SharedNtmbs &rhs){
	return lhs < rhs.get();
}

inline bool operator>(const SharedNtmbs &lhs, const SharedNtmbs &rhs){
	return std::strcmp(lhs.get(), rhs.get()) > 0;
}
inline bool operator>(const SharedNtmbs &lhs, const char *rhs){
	return std::strcmp(lhs.get(), rhs) > 0;
}
inline bool operator>(const SharedNtmbs &lhs, const std::string &rhs){
	return lhs.get() > rhs;
}
inline bool operator>(const char *lhs, const SharedNtmbs &rhs){
	return std::strcmp(lhs, rhs.get()) > 0;
}
inline bool operator>(const std::string &lhs, const SharedNtmbs &rhs){
	return lhs > rhs.get();
}

inline bool operator<=(const SharedNtmbs &lhs, const SharedNtmbs &rhs){
	return std::strcmp(lhs.get(), rhs.get()) <= 0;
}
inline bool operator<=(const SharedNtmbs &lhs, const char *rhs){
	return std::strcmp(lhs.get(), rhs) <= 0;
}
inline bool operator<=(const SharedNtmbs &lhs, const std::string &rhs){
	return lhs.get() <= rhs;
}
inline bool operator<=(const char *lhs, const SharedNtmbs &rhs){
	return std::strcmp(lhs, rhs.get()) <= 0;
}
inline bool operator<=(const std::string &lhs, const SharedNtmbs &rhs){
	return lhs <= rhs.get();
}

inline bool operator>=(const SharedNtmbs &lhs, const SharedNtmbs &rhs){
	return std::strcmp(lhs.get(), rhs.get()) >= 0;
}
inline bool operator>=(const SharedNtmbs &lhs, const char *rhs){
	return std::strcmp(lhs.get(), rhs) >= 0;
}
inline bool operator>=(const SharedNtmbs &lhs, const std::string &rhs){
	return lhs.get() >= rhs;
}
inline bool operator>=(const char *lhs, const SharedNtmbs &rhs){
	return std::strcmp(lhs, rhs.get()) >= 0;
}
inline bool operator>=(const std::string &lhs, const SharedNtmbs &rhs){
	return lhs >= rhs.get();
}

inline void swap(SharedNtmbs &lhs, SharedNtmbs &rhs) NOEXCEPT {
	lhs.swap(rhs);
}

extern std::ostream &operator<<(std::ostream &os, const SharedNtmbs &rhs);
extern std::wostream &operator<<(std::wostream &os, const SharedNtmbs &rhs);

}

#endif
