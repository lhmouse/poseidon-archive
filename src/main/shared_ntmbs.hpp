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
	typedef boost::shared_ptr<const char> NtmbsPtr;

private:
	NtmbsPtr m_ptr;

public:
	SharedNtmbs()
		: m_ptr(boost::shared_ptr<void>(), "")
	{
	}
	SharedNtmbs(const char *str, bool owning = false)
		: m_ptr(boost::shared_ptr<void>(), str)
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
	SharedNtmbs(const SharedNtmbs &rhs, bool owning = false)
		: m_ptr(rhs.m_ptr)
	{
		if(owning){
			forkOwning();
		}
	}
	SharedNtmbs(Move<SharedNtmbs> rhs, bool owning){
		rhs.swap(*this);
		if(owning){
			forkOwning();
		}
	}
#ifdef POSEIDON_CXX11
	SharedNtmbs(SharedNtmbs &&rhs) noexcept {
		rhs.swap(*this);
	}
#endif
	SharedNtmbs &operator=(const SharedNtmbs &rhs){
		m_ptr = rhs.m_ptr;
		return *this;
	}
	SharedNtmbs &operator=(Move<SharedNtmbs> rhs) NOEXCEPT {
		rhs.swap(*this);
		return *this;
	}

public:
	const char *get() const {
		return m_ptr.get();
	}

	bool isOwning() const;
	void forkOwning();

	void swap(SharedNtmbs &rhs) NOEXCEPT {
		m_ptr.swap(rhs.m_ptr);
	}

public:
	operator const char *() const {
		return get();
	}
	operator std::string() const {
		return std::string(get());
	}
};

static inline bool operator==(const SharedNtmbs &lhs, const SharedNtmbs &rhs){
	return std::strcmp(lhs.get(), rhs.get()) == 0;
}
static inline bool operator==(const SharedNtmbs &lhs, const char *rhs){
	return std::strcmp(lhs.get(), rhs) == 0;
}
static inline bool operator==(const SharedNtmbs &lhs, const std::string &rhs){
	return lhs.get() == rhs;
}
static inline bool operator==(const char *lhs, const SharedNtmbs &rhs){
	return std::strcmp(lhs, rhs.get()) == 0;
}
static inline bool operator==(const std::string &lhs, const SharedNtmbs &rhs){
	return lhs == rhs.get();
}

static inline bool operator!=(const SharedNtmbs &lhs, const SharedNtmbs &rhs){
	return std::strcmp(lhs.get(), rhs.get()) != 0;
}
static inline bool operator!=(const SharedNtmbs &lhs, const char *rhs){
	return std::strcmp(lhs.get(), rhs) != 0;
}
static inline bool operator!=(const SharedNtmbs &lhs, const std::string &rhs){
	return lhs.get() != rhs;
}
static inline bool operator!=(const char *lhs, const SharedNtmbs &rhs){
	return std::strcmp(lhs, rhs.get()) != 0;
}
static inline bool operator!=(const std::string &lhs, const SharedNtmbs &rhs){
	return lhs != rhs.get();
}

static inline bool operator<(const SharedNtmbs &lhs, const SharedNtmbs &rhs){
	return std::strcmp(lhs.get(), rhs.get()) < 0;
}
static inline bool operator<(const SharedNtmbs &lhs, const char *rhs){
	return std::strcmp(lhs.get(), rhs) < 0;
}
static inline bool operator<(const SharedNtmbs &lhs, const std::string &rhs){
	return lhs.get() < rhs;
}
static inline bool operator<(const char *lhs, const SharedNtmbs &rhs){
	return std::strcmp(lhs, rhs.get()) < 0;
}
static inline bool operator<(const std::string &lhs, const SharedNtmbs &rhs){
	return lhs < rhs.get();
}

static inline bool operator>(const SharedNtmbs &lhs, const SharedNtmbs &rhs){
	return std::strcmp(lhs.get(), rhs.get()) > 0;
}
static inline bool operator>(const SharedNtmbs &lhs, const char *rhs){
	return std::strcmp(lhs.get(), rhs) > 0;
}
static inline bool operator>(const SharedNtmbs &lhs, const std::string &rhs){
	return lhs.get() > rhs;
}
static inline bool operator>(const char *lhs, const SharedNtmbs &rhs){
	return std::strcmp(lhs, rhs.get()) > 0;
}
static inline bool operator>(const std::string &lhs, const SharedNtmbs &rhs){
	return lhs > rhs.get();
}

static inline bool operator<=(const SharedNtmbs &lhs, const SharedNtmbs &rhs){
	return std::strcmp(lhs.get(), rhs.get()) <= 0;
}
static inline bool operator<=(const SharedNtmbs &lhs, const char *rhs){
	return std::strcmp(lhs.get(), rhs) <= 0;
}
static inline bool operator<=(const SharedNtmbs &lhs, const std::string &rhs){
	return lhs.get() <= rhs;
}
static inline bool operator<=(const char *lhs, const SharedNtmbs &rhs){
	return std::strcmp(lhs, rhs.get()) <= 0;
}
static inline bool operator<=(const std::string &lhs, const SharedNtmbs &rhs){
	return lhs <= rhs.get();
}

static inline bool operator>=(const SharedNtmbs &lhs, const SharedNtmbs &rhs){
	return std::strcmp(lhs.get(), rhs.get()) >= 0;
}
static inline bool operator>=(const SharedNtmbs &lhs, const char *rhs){
	return std::strcmp(lhs.get(), rhs) >= 0;
}
static inline bool operator>=(const SharedNtmbs &lhs, const std::string &rhs){
	return lhs.get() >= rhs;
}
static inline bool operator>=(const char *lhs, const SharedNtmbs &rhs){
	return std::strcmp(lhs, rhs.get()) >= 0;
}
static inline bool operator>=(const std::string &lhs, const SharedNtmbs &rhs){
	return lhs >= rhs.get();
}

static inline void swap(SharedNtmbs &lhs, SharedNtmbs &rhs) NOEXCEPT {
	lhs.swap(rhs);
}

extern std::ostream &operator<<(std::ostream &os, const SharedNtmbs &rhs);
extern std::wostream &operator<<(std::wostream &os, const SharedNtmbs &rhs);

}

#endif
