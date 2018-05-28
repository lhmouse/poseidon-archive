// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_SESSION_BASE_HPP_
#define POSEIDON_SESSION_BASE_HPP_

#include <cstddef>
#include "stream_buffer.hpp"
#include "virtual_shared_from_this.hpp"

namespace Poseidon {

class Stream_buffer;

class Session_base : public virtual Virtual_shared_from_this {
public:
	Session_base() noexcept = default;
	~Session_base();

	Session_base(const Session_base &) = delete;
	Session_base &operator=(const Session_base &) = delete;

protected:
	virtual void on_connect() = 0;
	virtual void on_read_hup() = 0;
	virtual void on_close(int err_code) = 0;
	// 有数据可读触发回调，size 始终不为零。
	virtual void on_receive(Stream_buffer data) = 0;

	virtual bool send(Stream_buffer buffer) = 0;

public:
	virtual bool has_been_shutdown_read() const NOEXCEPT = 0;
	virtual bool has_been_shutdown_write() const NOEXCEPT = 0;
	virtual bool shutdown_read() NOEXCEPT = 0;
	virtual bool shutdown_write() NOEXCEPT = 0;
	virtual void force_shutdown() NOEXCEPT = 0;
};

}

#endif
