#ifndef POSEIDON_SHARED_NTMBS_HPP_
#define POSEIDON_SHARED_NTMBS_HPP_

#include "../cxx_ver.hpp"
#include <cstddef>
#include <cstring>
#include <boost/shared_ptr.hpp>
#include <boost/smart_ptr/owner_less.hpp>

namespace Poseidon {

class SharedNtmbs {
private:
	typedef boost::shared_ptr<const char> NtmbsPtr;

public:
	static SharedNtmbs createOwning(const char *str, std::size_t len);

	static SharedNtmbs createOwning(const char *str){
		return createOwning(str, std::strlen(str));
	}
	static SharedNtmbs createOwning(const std::string &str){
		return createOwning(str.data(), str.size());
	}

	static SharedNtmbs createNonOwning(const char *str){
		return SharedNtmbs(NtmbsPtr(boost::shared_ptr<void>(), str));
	}
	static SharedNtmbs createNonOwning(const std::string &str){
		return createNonOwning(str.c_str());
	}

private:
	NtmbsPtr m_ptr;

private:
	explicit SharedNtmbs(NtmbsPtr ptr) NOEXCEPT
		: m_ptr(STD_MOVE(ptr))
	{
	}

public:
	SharedNtmbs() NOEXCEPT
		: m_ptr(boost::shared_ptr<void>(), "")
	{
	}

public:
	const char *get() const {
		return m_ptr.get();
	}

	bool isOwning() const {
		typedef boost::owner_less<NtmbsPtr> OwnerLess;
		return OwnerLess()(m_ptr, NtmbsPtr()) || OwnerLess()(NtmbsPtr(), m_ptr);
	}
	SharedNtmbs forkOwning() const {
		if(isOwning()){
			return *this;
		}
		return createOwning(get());
	}

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
static inline bool operator==(const char *lhs, const SharedNtmbs &rhs){
	return std::strcmp(lhs, rhs.get()) == 0;
}

static inline bool operator!=(const SharedNtmbs &lhs, const SharedNtmbs &rhs){
	return std::strcmp(lhs.get(), rhs.get()) != 0;
}
static inline bool operator!=(const SharedNtmbs &lhs, const char *rhs){
	return std::strcmp(lhs.get(), rhs) != 0;
}
static inline bool operator!=(const char *lhs, const SharedNtmbs &rhs){
	return std::strcmp(lhs, rhs.get()) != 0;
}

static inline bool operator<(const SharedNtmbs &lhs, const SharedNtmbs &rhs){
	return std::strcmp(lhs.get(), rhs.get()) < 0;
}
static inline bool operator<(const SharedNtmbs &lhs, const char *rhs){
	return std::strcmp(lhs.get(), rhs) < 0;
}
static inline bool operator<(const char *lhs, const SharedNtmbs &rhs){
	return std::strcmp(lhs, rhs.get()) < 0;
}

static inline bool operator>(const SharedNtmbs &lhs, const SharedNtmbs &rhs){
	return std::strcmp(lhs.get(), rhs.get()) > 0;
}
static inline bool operator>(const SharedNtmbs &lhs, const char *rhs){
	return std::strcmp(lhs.get(), rhs) > 0;
}
static inline bool operator>(const char *lhs, const SharedNtmbs &rhs){
	return std::strcmp(lhs, rhs.get()) > 0;
}

static inline bool operator<=(const SharedNtmbs &lhs, const SharedNtmbs &rhs){
	return std::strcmp(lhs.get(), rhs.get()) <= 0;
}
static inline bool operator<=(const SharedNtmbs &lhs, const char *rhs){
	return std::strcmp(lhs.get(), rhs) <= 0;
}
static inline bool operator<=(const char *lhs, const SharedNtmbs &rhs){
	return std::strcmp(lhs, rhs.get()) <= 0;
}

static inline bool operator>=(const SharedNtmbs &lhs, const SharedNtmbs &rhs){
	return std::strcmp(lhs.get(), rhs.get()) >= 0;
}
static inline bool operator>=(const SharedNtmbs &lhs, const char *rhs){
	return std::strcmp(lhs.get(), rhs) >= 0;
}
static inline bool operator>=(const char *lhs, const SharedNtmbs &rhs){
	return std::strcmp(lhs, rhs.get()) >= 0;
}

static inline void swap(SharedNtmbs &lhs, SharedNtmbs &rhs) NOEXCEPT {
	lhs.swap(rhs);
}

}

#endif
