// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_HTTP_UPGRADED_SESSION_BASE_HPP_
#define POSEIDON_HTTP_UPGRADED_SESSION_BASE_HPP_

#include "../session_base.hpp"
#include <boost/shared_ptr.hpp>
#include <boost/weak_ptr.hpp>

namespace Poseidon {

namespace Http {
	class Session;

	class UpgradedSessionBase : public SessionBase {
		friend Session;

	private:
		const boost::weak_ptr<Session> m_parent;

	public:
		UpgradedSessionBase(const boost::shared_ptr<Session> &parent);
		~UpgradedSessionBase();

	protected:
		void on_connect() OVERRIDE;
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

		boost::weak_ptr<const Session> get_weak_parent() const {
			return m_parent;
		}
		boost::weak_ptr<Session> get_weak_parent(){
			return m_parent;
		}

		boost::shared_ptr<const Session> get_parent() const {
			return m_parent.lock();
		}
		boost::shared_ptr<Session> get_parent(){
			return m_parent.lock();
		}

		// 以下所有函数，如果原来的 Session 被删除，抛出 bad_weak_ptr。
		boost::shared_ptr<const Session> get_safe_parent() const {
			return boost::shared_ptr<const Session>(m_parent);
		}
		boost::shared_ptr<Session> get_safe_parent(){
			return boost::shared_ptr<Session>(m_parent);
		}

		void set_timeout(boost::uint64_t timeout);
	};
}

}

#endif
