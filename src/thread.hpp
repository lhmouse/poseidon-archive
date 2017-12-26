// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2017, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_THREAD_HPP_
#define POSEIDON_THREAD_HPP_

#include "cxx_ver.hpp"
#include "cxx_util.hpp"
#include <boost/shared_ptr.hpp>
#include <boost/function.hpp>

namespace Poseidon {

class Thread : NONCOPYABLE {
private:
	class Impl;
	boost::shared_ptr<Impl> m_impl;

public:
	Thread() NOEXCEPT;
	Thread(boost::function<void ()> proc, const char *tag); // tag 用于在日志中显示。最多四个字符。
	Thread(Move<Thread> rhs) NOEXCEPT
		: m_impl()
	{
		rhs.swap(*this);
	}
	Thread &operator=(Move<Thread> rhs) NOEXCEPT {
		Thread(STD_MOVE(rhs)).swap(*this);
		return *this;
	}
	~Thread(); // if(joinable()){ std::terminate(); }

public:
	void swap(Thread &rhs) NOEXCEPT {
		using std::swap;
		swap(m_impl, rhs.m_impl);
	}

	bool joinable() const NOEXCEPT;
	void join();
//	void detach();
};

inline void swap(Thread &lhs, Thread &rhs) NOEXCEPT {
	lhs.swap(rhs);
}

}

#endif
