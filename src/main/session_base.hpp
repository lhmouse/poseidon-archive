// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014, LH_Mouse. All wrongs reserved.

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
	// 不要不写析构函数，否则 RTTI 将无法在动态库中使用。
	~SessionBase();

private:
	// 有数据可读触发回调，size 始终不为零。
	virtual void onReadAvail(const void *data, std::size_t size) = 0;

public:
	// 在使用 final = true 调用 send() 或调用 forceShutdown() 之后返回 true。
	virtual bool hasBeenShutdown() const = 0;
	// 若 final = true，调用后 onReadAvail() 将不会被触发，
	// 此后任何 send() 将不会进行任何操作。
	// 套接字将会在未发送的数据被全部发送之后被正常关闭。
	virtual bool send(StreamBuffer buffer, bool final = false) = 0;
	// 强行关闭会话以及套接字，未发送数据丢失。
	virtual bool forceShutdown() = 0;
};

}

#endif
