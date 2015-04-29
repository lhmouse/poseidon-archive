// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_MUTEX_HPP_
#define POSEIDON_MUTEX_HPP_

#include "cxx_util.hpp"
#include <boost/scoped_ptr.hpp>

namespace Poseidon {

class ConditionVariable;

class Mutex : NONCOPYABLE {
	friend ConditionVariable;

private:
	class Impl; // pthread_mutex_t

public:
	class ScopedLock : NONCOPYABLE {
		friend ConditionVariable;

	private:
		Mutex *m_owner;
		bool m_locked;

	public:
		ScopedLock();
		explicit ScopedLock(Mutex &owner, bool locksOwner = true);
		~ScopedLock();

	public:
		bool locked() const NOEXCEPT;
		void lock() NOEXCEPT;
		void unlock() NOEXCEPT;

		void swap(ScopedLock &rhs) NOEXCEPT;
	};

private:
	const boost::scoped_ptr<Impl> m_impl;

public:
	explicit Mutex(bool recursive = false);
	~Mutex();
};

inline void swap(Mutex::ScopedLock &lhs, Mutex::ScopedLock &rhs) NOEXCEPT {
	lhs.swap(rhs);
}

}

#endif
