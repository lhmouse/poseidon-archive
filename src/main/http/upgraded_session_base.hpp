// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_HTTP_UPGRADED_SESSION_BASE_HPP_
#define POSEIDON_HTTP_UPGRADED_SESSION_BASE_HPP_

#include "../session_base.hpp"
#include <string>
#include <boost/shared_ptr.hpp>
#include <boost/weak_ptr.hpp>
#include <boost/function.hpp>

namespace Poseidon {

class OptionalMap;

namespace Http {
	class Session;

	class UpgradedSessionBase : public SessionBase {
		friend Session;

	private:
		const boost::weak_ptr<Session> m_parent;
		const std::string m_uri;

	protected:
		UpgradedSessionBase(const boost::shared_ptr<Session> &parent, std::string uri);

	private:
		virtual void onReadAvail(const void *data, std::size_t size) = 0;
		virtual void onClose() NOEXCEPT;
		virtual void onReadHup() NOEXCEPT;

	public:
		bool send(StreamBuffer buffer, bool fin = false) FINAL;

		bool hasBeenShutdown() const OVERRIDE;
		bool shutdown() NOEXCEPT;
		bool forceShutdown() NOEXCEPT OVERRIDE;

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

		const std::string &getUri() const {
			return m_uri;
		}

		void setTimeout(boost::uint64_t timeout);
		void registerOnClose(boost::function<void ()> callback);
	};
}

}

#endif
