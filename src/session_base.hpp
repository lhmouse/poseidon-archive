// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_SESSION_BASE_HPP_
#define POSEIDON_SESSION_BASE_HPP_

#include "cxx_util.hpp"
#include <cstddef>
#include "virtual_shared_from_this.hpp"

namespace Poseidon {

class StreamBuffer;

class SessionBase : NONCOPYABLE, public virtual VirtualSharedFromThis {
public:
	// 不要不写析构函数，否则 RTTI 将无法在动态库中使用。
	~SessionBase();

protected:
	// 有数据可读触发回调，size 始终不为零。
	virtual void onReadAvail(const void *data, std::size_t size) = 0;

	virtual void onReadHup() NOEXCEPT = 0;
	virtual void onWriteHup() NOEXCEPT = 0;
	virtual void onClose(int errCode) NOEXCEPT = 0;

public:
	virtual bool send(StreamBuffer buffer) = 0;

	virtual bool hasBeenShutdownRead() const NOEXCEPT = 0;
	virtual bool hasBeenShutdownWrite() const NOEXCEPT = 0;
	virtual bool shutdownRead() NOEXCEPT = 0;
	virtual bool shutdownWrite() NOEXCEPT = 0;
	virtual void forceShutdown() NOEXCEPT = 0;
};

}

#endif
