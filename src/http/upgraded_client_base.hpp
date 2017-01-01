// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2017, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_HTTP_UPGRADED_CLIENT_BASE_HPP_
#define POSEIDON_HTTP_UPGRADED_CLIENT_BASE_HPP_

#include "../session_base.hpp"
#include <boost/shared_ptr.hpp>
#include <boost/weak_ptr.hpp>

namespace Poseidon {

namespace Http {
	class LowLevelClient;

	class UpgradedClientBase : public SessionBase {
		friend LowLevelClient;

	private:
		const boost::weak_ptr<LowLevelClient> m_parent;

	public:
		UpgradedClientBase(const boost::shared_ptr<LowLevelClient> &parent);
		~UpgradedClientBase();

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

		boost::weak_ptr<const LowLevelClient> get_weak_parent() const {
			return m_parent;
		}
		boost::weak_ptr<LowLevelClient> get_weak_parent(){
			return m_parent;
		}

		boost::shared_ptr<const LowLevelClient> get_parent() const {
			return m_parent.lock();
		}
		boost::shared_ptr<LowLevelClient> get_parent(){
			return m_parent.lock();
		}

		// 以下所有函数，如果原来的 LowLevelClient 被删除，抛出 bad_weak_ptr。
		boost::shared_ptr<const LowLevelClient> get_safe_parent() const {
			return boost::shared_ptr<const LowLevelClient>(m_parent);
		}
		boost::shared_ptr<LowLevelClient> get_safe_parent(){
			return boost::shared_ptr<LowLevelClient>(m_parent);
		}
	};
}

}

#endif
