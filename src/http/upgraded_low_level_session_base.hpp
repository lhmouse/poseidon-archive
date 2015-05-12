// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_HTTP_UPGRADED_LOW_LEVEL_SESSION_BASE_HPP_
#define POSEIDON_HTTP_UPGRADED_LOW_LEVEL_SESSION_BASE_HPP_

#include "../session_base.hpp"
#include <string>
#include <boost/shared_ptr.hpp>
#include <boost/weak_ptr.hpp>

namespace Poseidon {

namespace Http {
	class RequestHeaders;
	class LowLevelSession;

	class UpgradedLowLevelSessionBase : public SessionBase {
		friend LowLevelSession;

	private:
		const boost::weak_ptr<LowLevelSession> m_parent;
		const std::string m_uri;

	protected:
		UpgradedLowLevelSessionBase(const boost::shared_ptr<LowLevelSession> &parent, std::string uri);
		~UpgradedLowLevelSessionBase();

	private:
		void onReadHup() NOEXCEPT OVERRIDE;
		void onWriteHup() NOEXCEPT OVERRIDE;
		void onClose(int errCode) NOEXCEPT OVERRIDE;

		virtual void onInit(RequestHeaders requestHeaders,
			std::vector<std::string> transferEncoding, StreamBuffer entity);
		virtual void onReadAvail(const void *data, std::size_t size) = 0;

	public:
		bool send(StreamBuffer buffer) OVERRIDE;

		bool hasBeenShutdownRead() const NOEXCEPT OVERRIDE;
		bool shutdownRead() NOEXCEPT;
		bool hasBeenShutdownWrite() const NOEXCEPT OVERRIDE;
		bool shutdownWrite() NOEXCEPT;
		void forceShutdown() NOEXCEPT;

		boost::weak_ptr<const LowLevelSession> getWeakParent() const {
			return m_parent;
		}
		boost::weak_ptr<LowLevelSession> getWeakParent(){
			return m_parent;
		}

		boost::shared_ptr<const LowLevelSession> getParent() const {
			return m_parent.lock();
		}
		boost::shared_ptr<LowLevelSession> getParent(){
			return m_parent.lock();
		}

		// 以下所有函数，如果原来的 LowLevelSession 被删除，抛出 bad_weak_ptr。
		boost::shared_ptr<const LowLevelSession> getSafeParent() const {
			return boost::shared_ptr<const LowLevelSession>(m_parent);
		}
		boost::shared_ptr<LowLevelSession> getSafeParent(){
			return boost::shared_ptr<LowLevelSession>(m_parent);
		}

		const std::string &getUri() const {
			return m_uri;
		}

		void setTimeout(boost::uint64_t timeout);
	};
}

}

#endif
