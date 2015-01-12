// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_EPOLL_HPP_
#define POSEIDON_EPOLL_HPP_

#include "cxx_util.hpp"
#include "raii.hpp"
#include <boost/thread/mutex.hpp>
#include <boost/scoped_ptr.hpp>
#include <boost/shared_ptr.hpp>
#include <vector>
#include <cstddef>

namespace Poseidon {

class TcpSessionBase;

class Epoll : NONCOPYABLE {
	friend TcpSessionBase;

private:
	class SessionMapImpl;

private:
	mutable boost::mutex m_mutex;
	UniqueFile m_epoll;
	boost::scoped_ptr<SessionMapImpl> m_sessions;

public:
	Epoll();
	~Epoll();

private:
	void notifyWriteable(TcpSessionBase *session);

public:
	void addSession(const boost::shared_ptr<TcpSessionBase> &session);
	void removeSession(const boost::shared_ptr<TcpSessionBase> &session);
	void snapshot(std::vector<boost::shared_ptr<TcpSessionBase> > &sessions) const;
	void clear();

	// 这三个函数必须位于同一个线程内调用。
	std::size_t wait(unsigned timeout);
	std::size_t pumpReadable();
	std::size_t pumpWriteable();
};

}

#endif
