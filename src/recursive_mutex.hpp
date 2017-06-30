// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2017, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_RECURSIVE_MUTEX_HPP_
#define POSEIDON_RECURSIVE_MUTEX_HPP_

#include "cxx_ver.hpp"
#include "cxx_util.hpp"
#include <pthread.h>

namespace Poseidon {

class RecursiveMutex : NONCOPYABLE {
public:
	class UniqueLock : NONCOPYABLE {
	private:
		RecursiveMutex *m_target;
		bool m_locked;

	private:
		UniqueLock(const UniqueLock &rhs);
		UniqueLock &operator=(const UniqueLock &rhs);

	public:
		UniqueLock();
		explicit UniqueLock(RecursiveMutex &target, bool locks_target = true);
		UniqueLock(Move<UniqueLock> rhs) NOEXCEPT
			: m_target(NULLPTR), m_locked(false)
		{
			rhs.swap(*this);
		}
		UniqueLock &operator=(Move<UniqueLock> rhs) NOEXCEPT {
			UniqueLock(STD_MOVE(rhs)).swap(*this);
			return *this;
		}
		~UniqueLock();

	public:
		bool is_locked() const NOEXCEPT;
		void lock() NOEXCEPT;
		void unlock() NOEXCEPT;

		void swap(UniqueLock &rhs) NOEXCEPT {
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
		typedef bool (UniqueLock::*DummyBool_)() const;
		operator DummyBool_() const NOEXCEPT {
			return is_locked() ? &UniqueLock::is_locked : 0;
		}
#endif
	};

private:
	::pthread_mutex_t m_mutex;

public:
	RecursiveMutex();
	~RecursiveMutex();
};

inline void swap(RecursiveMutex::UniqueLock &lhs, RecursiveMutex::UniqueLock &rhs) NOEXCEPT {
	lhs.swap(rhs);
}

}

#endif
