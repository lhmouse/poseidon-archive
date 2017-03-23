// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2017, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_SESSION_BASE_HPP_
#define POSEIDON_SESSION_BASE_HPP_

#include "cxx_util.hpp"
#include <cstddef>
#include "stream_buffer.hpp"
#include "virtual_shared_from_this.hpp"

namespace Poseidon {

class StreamBuffer;

class SessionBase : NONCOPYABLE, public virtual VirtualSharedFromThis {
public:
	// 不要不写析构函数，否则 RTTI 将无法在动态库中使用。
	~SessionBase();

protected:
	virtual void on_connect() = 0;
	virtual void on_read_hup() = 0;
	virtual void on_close(int err_code) NOEXCEPT = 0;
	// 有数据可读触发回调，size 始终不为零。
	virtual void on_receive(StreamBuffer data) = 0;

	virtual bool send(StreamBuffer buffer) = 0;

public:
	virtual bool has_been_shutdown_read() const NOEXCEPT = 0;
	virtual bool has_been_shutdown_write() const NOEXCEPT = 0;
	virtual bool shutdown_read() NOEXCEPT = 0;
	virtual bool shutdown_write() NOEXCEPT = 0;
	virtual void force_shutdown() NOEXCEPT = 0;
};

}

#endif
