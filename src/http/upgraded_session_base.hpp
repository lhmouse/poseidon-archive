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
		void onReadHup() NOEXCEPT OVERRIDE;
		void onClose(int errCode) NOEXCEPT OVERRIDE;

		virtual void onReadAvail(const void *data, std::size_t size) = 0;

	public:
		bool send(StreamBuffer buffer) OVERRIDE;

		bool hasBeenShutdownRead() const NOEXCEPT OVERRIDE;
		bool shutdownRead() NOEXCEPT;
		bool hasBeenShutdownWrite() const NOEXCEPT OVERRIDE;
		bool shutdownWrite() NOEXCEPT;
		void forceShutdown() NOEXCEPT;

		boost::weak_ptr<const Session> getWeakParent() const {
			return m_parent;
		}
		boost::weak_ptr<Session> getWeakParent(){
			return m_parent;
		}

		boost::shared_ptr<const Session> getParent() const {
			return m_parent.lock();
		}
		boost::shared_ptr<Session> getParent(){
			return m_parent.lock();
		}

		// 以下所有函数，如果原来的 Session 被删除，抛出 bad_weak_ptr。
		boost::shared_ptr<const Session> getSafeParent() const {
			return boost::shared_ptr<const Session>(m_parent);
		}
		boost::shared_ptr<Session> getSafeParent(){
			return boost::shared_ptr<Session>(m_parent);
		}

		void setTimeout(boost::uint64_t timeout);
	};
}

}

#endif
