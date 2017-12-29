// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2017, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_HTTP_UPGRADED_SESSION_BASE_HPP_
#define POSEIDON_HTTP_UPGRADED_SESSION_BASE_HPP_

#include "../session_base.hpp"
#include "../fwd.hpp"
#include "../ip_port.hpp"
#include <boost/shared_ptr.hpp>
#include <boost/weak_ptr.hpp>
#include <boost/cstdint.hpp>

namespace Poseidon {
namespace Http {

class LowLevelSession;
class LowLevelClient;

class UpgradedSessionBase : public SessionBase {
	friend LowLevelSession;
	friend LowLevelClient;

private:
	const boost::weak_ptr<TcpSessionBase> m_parent;

public:
	UpgradedSessionBase(const boost::shared_ptr<TcpSessionBase> &parent);
	~UpgradedSessionBase();

protected:
	void on_connect() OVERRIDE = 0;
	void on_read_hup() OVERRIDE = 0;
	void on_close(int err_code) OVERRIDE = 0;
	void on_receive(StreamBuffer data) OVERRIDE = 0;

	virtual void on_shutdown_timer(boost::uint64_t now);

public:
	bool has_been_shutdown_read() const NOEXCEPT OVERRIDE;
	bool shutdown_read() NOEXCEPT OVERRIDE;
	bool has_been_shutdown_write() const NOEXCEPT OVERRIDE;
	bool shutdown_write() NOEXCEPT OVERRIDE;
	void force_shutdown() NOEXCEPT OVERRIDE;

	boost::weak_ptr<const TcpSessionBase> get_weak_parent() const {
		return m_parent;
	}
	boost::weak_ptr<TcpSessionBase> get_weak_parent(){
		return m_parent;
	}

	boost::shared_ptr<const TcpSessionBase> get_parent() const {
		return m_parent.lock();
	}
	boost::shared_ptr<TcpSessionBase> get_parent(){
		return m_parent.lock();
	}

	const IpPort &get_remote_info() const NOEXCEPT;
	const IpPort &get_local_info() const NOEXCEPT;

	bool is_throttled() const;

	void set_no_delay(bool enabled = true);
	void set_timeout(boost::uint64_t timeout);

	bool send(StreamBuffer buffer) OVERRIDE;
};

}
}

#endif
