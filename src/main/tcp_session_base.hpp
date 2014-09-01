#ifndef POSEIDON_TCP_SESSION_BASE_HPP_
#define POSEIDON_TCP_SESSION_BASE_HPP_

#include <string>
#include <cstddef>
#include <boost/noncopyable.hpp>
#include <boost/thread/mutex.hpp>
#include "virtual_shared_from_this.hpp"
#include "raii.hpp"
#include "atomic.hpp"
#include "stream_buffer.hpp"

namespace Poseidon {

class TcpSessionBase : boost::noncopyable
	, public virtual VirtualSharedFromThis
{
private:
	ScopedFile m_socket;
	std::string m_remoteIp;

	volatile bool m_readShutdown;
	volatile bool m_shutdown;
	mutable boost::mutex m_bufferMutex;
	StreamBuffer m_sendBuffer;

protected:
	explicit TcpSessionBase(ScopedFile &socket);
	virtual ~TcpSessionBase();

public:
	int getFd() const {
		return m_socket.get();
	}
	const std::string &getRemoteIp() const {
		return m_remoteIp;
	}
	bool hasReadBeenShutdown() const {
		return atomicLoad(m_readShutdown);
	}
	bool hasBeenShutdown() const {
		return atomicLoad(m_shutdown);
	}

	// 如果 size 为零则返回所有待发送字节数。
	std::size_t peekWriteAvail(boost::mutex::scoped_lock &lock,
		void *data, std::size_t size) const;
	// 从队列中移除指定的字节数。
	void notifyWritten(std::size_t size);

	// 有数据可读触发回调，size 始终不为零。
	virtual void onReadAvail(const void *data, std::size_t size) = 0;

	// 这三个函数是线程安全的。
	void send(const void *data, std::size_t size);
	void send(const StreamBuffer &buffer);
#ifdef POSEIDON_CXX11
	void send(StreamBuffer &&buffer){
		sendWithMove(buffer);
	}
#endif
	// 执行后 buffer 置空。
	void sendWithMove(StreamBuffer &buffer);

	// 关闭读端，调用后 onReadAvail() 将不会被触发，但是依然可写。
	// 一旦无任何数据可写，该会话会被立即关闭。
	void shutdownRead();
	// 关闭会话，此后任何 send() 或 sendWithMove() 将抛出一个异常。
	// 套接字将会在未发送的数据被全部发送之后被正常关闭。
	void shutdown();
	// 强行关闭会话以及套接字，未发送数据丢失。
	void forceShutdown();
};

}

#endif
