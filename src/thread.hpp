// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_THREAD_HPP_
#define POSEIDON_THREAD_HPP_

#include "cxx_ver.hpp"
#include "cxx_util.hpp"
#include "shared_nts.hpp"
#include <boost/function.hpp>

namespace Poseidon {

class Thread : NONCOPYABLE {
private:
	boost::shared_ptr<void> m_tcb;

public:
	Thread() NOEXCEPT
		: m_tcb()
	{
		//
	}
	Thread(boost::function<void ()> proc, SharedNts tag, SharedNts name);
	~Thread(); // The destructor calls `std::terminate()` if `joinable()` returns `true`.

public:
	bool joinable() const NOEXCEPT;
	void join();

	void swap(Thread &rhs) NOEXCEPT {
		using std::swap;
		swap(m_tcb, rhs.m_tcb);
	}
};

inline void swap(Thread &lhs, Thread &rhs) NOEXCEPT {
	lhs.swap(rhs);
}

}

#endif
