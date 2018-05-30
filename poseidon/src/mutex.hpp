// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_MUTEX_HPP_
#define POSEIDON_MUTEX_HPP_

#include "cxx_ver.hpp"
#include "cxx_util.hpp"
#include <pthread.h>

namespace Poseidon {

class Condition_variable;

class Mutex : NONCOPYABLE {
	friend Condition_variable;

public:
	class Unique_lock;

private:
	::pthread_mutex_t m_mutex;

public:
	Mutex();
	~Mutex();
};

class Mutex::Unique_lock : NONCOPYABLE {
	friend Condition_variable;

private:
	Mutex *m_target;
	bool m_locked;

private:
	Unique_lock(const Unique_lock &rhs);
	Unique_lock & operator=(const Unique_lock &rhs);

public:
	Unique_lock();
	explicit Unique_lock(Mutex &target, bool locks_target = true);
	Unique_lock(Move<Unique_lock> rhs) NOEXCEPT
		: m_target(NULLPTR), m_locked(false)
	{
		rhs.swap(*this);
	}
	Unique_lock & operator=(Move<Unique_lock> rhs) NOEXCEPT {
		Unique_lock(STD_MOVE(rhs)).swap(*this);
		return *this;
	}
	~Unique_lock();

public:
	bool is_locked() const NOEXCEPT;
	void lock() NOEXCEPT;
	void unlock() NOEXCEPT;

	void swap(Unique_lock &rhs) NOEXCEPT {
		using std::swap;
		swap(m_target, rhs.m_target);
		swap(m_locked, rhs.m_locked);
	}

public:
#ifdef POSEIDON_CXX11
	explicit operator bool() const noexcept {
		return is_locked();
	}
#else
	typedef bool (Unique_lock::*Dummy_bool_)() const;
	operator Dummy_bool_() const NOEXCEPT {
		return is_locked() ? &Unique_lock::is_locked : 0;
	}
#endif
};

inline void swap(Mutex::Unique_lock &lhs, Mutex::Unique_lock &rhs) NOEXCEPT {
	lhs.swap(rhs);
}

}

#endif
