// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2016, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_EPOLL_HPP_
#define POSEIDON_EPOLL_HPP_

#include "cxx_util.hpp"
#include <boost/scoped_ptr.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/enable_shared_from_this.hpp>
#include <vector>
#include <cstddef>
#include "raii.hpp"
#include "recursive_mutex.hpp"

namespace Poseidon {

class TcpSessionBase;

class Epoll : NONCOPYABLE, public boost::enable_shared_from_this<Epoll> {
	friend TcpSessionBase;

public: // XXX
	class SessionMap;

private:
	mutable RecursiveMutex m_mutex;
	UniqueFile m_epoll;
	boost::scoped_ptr<SessionMap> m_sessions;

public:
	Epoll();
	~Epoll();

private:
	void notify_writeable(TcpSessionBase *session) NOEXCEPT;
	void notify_unlinked(TcpSessionBase *session) NOEXCEPT;

public:
	void add_session(const boost::shared_ptr<TcpSessionBase> &session);
	void remove_session(const boost::shared_ptr<TcpSessionBase> &session);
	void snapshot(std::vector<boost::shared_ptr<TcpSessionBase> > &sessions) const;
	void clear();

	// 这三个函数必须位于同一个线程内调用。
	std::size_t wait(unsigned timeout) NOEXCEPT;
	std::size_t pump_readable();
	std::size_t pump_writeable();
};

}

#endif
