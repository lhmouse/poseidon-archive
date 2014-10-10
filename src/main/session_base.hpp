#ifndef POSEIDON_SESSION_BASE_HPP_
#define POSEIDON_SESSION_BASE_HPP_

#include <cstddef>
#include <boost/noncopyable.hpp>
#include "virtual_shared_from_this.hpp"

namespace Poseidon {

class StreamBuffer;

class SessionBase : boost::noncopyable,
	public virtual VirtualSharedFromThis
{
public:
	// 实现定义。
	virtual const std::string &getRemoteIp() const = 0;
	// 有数据可读触发回调，size 始终不为零。
	virtual void onReadAvail(const void *data, std::size_t size) = 0;
	// 执行后 buffer 置空。这个函数是线程安全的。
	virtual bool send(StreamBuffer buffer) = 0;
	// 在调用 shutdown() 或 forceShutdown() 之后返回 true。
	virtual bool hasBeenShutdown() const = 0;
	// 调用后 onReadAvail() 将不会被触发，
	// 此后任何 send() 将不会进行任何操作。
	// 套接字将会在未发送的数据被全部发送之后被正常关闭。
	// 如果调用 shutdown() 之前套接字未被关闭则返回 true。
	virtual bool shutdown() = 0;
	// 强行关闭会话以及套接字，未发送数据丢失。
	virtual bool forceShutdown() = 0;
};

}

#endif
