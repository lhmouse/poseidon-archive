// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_HTTP_UPGRADED_SESSION_BASE_HPP_
#define POSEIDON_HTTP_UPGRADED_SESSION_BASE_HPP_

#include "../session_base.hpp"
#include "../tcp_session_base.hpp"
#include <boost/shared_ptr.hpp>
#include <boost/weak_ptr.hpp>
#include <boost/cstdint.hpp>

namespace Poseidon {
namespace Http {

class Low_level_session;
class Low_level_client;

class Upgraded_session_base : public Session_base {
	friend Low_level_session;
	friend Low_level_client;

private:
	const boost::weak_ptr<Tcp_session_base> m_parent;

public:
	Upgraded_session_base(const boost::shared_ptr<Tcp_session_base> &parent);
	~Upgraded_session_base();

protected:
	void on_connect() OVERRIDE = 0;
	void on_read_hup() OVERRIDE = 0;
	void on_close(int err_code) OVERRIDE = 0;
	void on_receive(Stream_buffer data) OVERRIDE = 0;

	virtual void on_shutdown_timer(boost::uint64_t now);

public:
	bool has_been_shutdown_read() const NOEXCEPT OVERRIDE;
	bool shutdown_read() NOEXCEPT OVERRIDE;
	bool has_been_shutdown_write() const NOEXCEPT OVERRIDE;
	bool shutdown_write() NOEXCEPT OVERRIDE;
	void force_shutdown() NOEXCEPT OVERRIDE;

	boost::weak_ptr<const Tcp_session_base> get_weak_parent() const {
		return m_parent;
	}
	boost::weak_ptr<Tcp_session_base> get_weak_parent(){
		return m_parent;
	}

	boost::shared_ptr<const Tcp_session_base> get_parent() const {
		return m_parent.lock();
	}
	boost::shared_ptr<Tcp_session_base> get_parent(){
		return m_parent.lock();
	}

	const Ip_port & get_remote_info() const NOEXCEPT;
	const Ip_port & get_local_info() const NOEXCEPT;

	bool is_throttled() const;

	void set_no_delay(bool enabled = true);
	void set_timeout(boost::uint64_t timeout);

	bool send(Stream_buffer buffer) OVERRIDE;
};

}
}

#endif
