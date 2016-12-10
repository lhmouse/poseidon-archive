// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2016, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_HTTP_UPGRADED_SESSION_BASE_HPP_
#define POSEIDON_HTTP_UPGRADED_SESSION_BASE_HPP_

#include "../session_base.hpp"
#include "../ip_port.hpp"
#include <boost/shared_ptr.hpp>
#include <boost/weak_ptr.hpp>

namespace Poseidon {

namespace Http {
	class LowLevelSession;

	class UpgradedSessionBase : public SessionBase {
		friend LowLevelSession;

	private:
		const boost::weak_ptr<LowLevelSession> m_parent;

	public:
		UpgradedSessionBase(const boost::shared_ptr<LowLevelSession> &parent);
		~UpgradedSessionBase();

	protected:
		void on_read_hup() NOEXCEPT OVERRIDE;
		void on_close(int err_code) NOEXCEPT OVERRIDE;

		virtual void on_read_avail(StreamBuffer data) = 0;

	public:
		bool send(StreamBuffer buffer) OVERRIDE;

		bool has_been_shutdown_read() const NOEXCEPT OVERRIDE;
		bool shutdown_read() NOEXCEPT;
		bool has_been_shutdown_write() const NOEXCEPT OVERRIDE;
		bool shutdown_write() NOEXCEPT;
		void force_shutdown() NOEXCEPT;

		boost::weak_ptr<const LowLevelSession> get_weak_parent() const {
			return m_parent;
		}
		boost::weak_ptr<LowLevelSession> get_weak_parent(){
			return m_parent;
		}

		boost::shared_ptr<const LowLevelSession> get_parent() const {
			return m_parent.lock();
		}
		boost::shared_ptr<LowLevelSession> get_parent(){
			return m_parent.lock();
		}

		// 以下所有函数，如果原来的 LowLevelSession 被删除，抛出 bad_weak_ptr。
		boost::shared_ptr<const LowLevelSession> get_safe_parent() const {
			return boost::shared_ptr<const LowLevelSession>(m_parent);
		}
		boost::shared_ptr<LowLevelSession> get_safe_parent(){
			return boost::shared_ptr<LowLevelSession>(m_parent);
		}

		const IpPort &get_remote_info() const;
		const IpPort &get_local_info() const;
		IpPort get_remote_info_nothrow() const NOEXCEPT;
		IpPort get_local_info_nothrow() const NOEXCEPT;

		void set_timeout(boost::uint64_t timeout);
	};
}

}

#endif
