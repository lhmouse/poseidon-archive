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
#ifdef POSEIDON_CXX11
	SharedNtmbs(std::string &&, bool owning = false) = delete;
#endif
	SharedNtmbs(const SharedNtmbs &rhs, bool owning)
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

public:
	const char *get() const {
		return m_ptr.get();
	}
	bool empty() const {
		return get()[0] == 0;
	}

	bool isOwning() const;
	void forkOwning();

	void swap(SharedNtmbs &rhs) NOEXCEPT {
		m_ptr.swap(rhs.m_ptr);
	}

public:
	const char &operator[](std::size_t index) const {
		return get()[index];
	}
#ifdef POSEIDON_CXX11
	explicit operator bool() const {
		return !empty();
	}
#else
	typedef const char *(SharedNtmbs::*DummyBool_)() const;
	operator DummyBool_() const {
		return !empty() ? &SharedNtmbs::get : 0;
	}
#endif
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
