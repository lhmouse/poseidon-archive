// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_HTTP_UPGRADED_SESSION_BASE_HPP_
#define POSEIDON_HTTP_UPGRADED_SESSION_BASE_HPP_

#include "../session_base.hpp"
#include <string>
#include <boost/shared_ptr.hpp>
#include <boost/weak_ptr.hpp>

namespace Poseidon {

class OptionalMap;
class HttpSession;

class HttpUpgradedSessionBase : public SessionBase {
	friend class HttpSession;

private:
	const boost::weak_ptr<HttpSession> m_parent;

protected:
	explicit HttpUpgradedSessionBase(const boost::shared_ptr<HttpSession> &parent);

private:
	virtual void onInitContents(const void *data, std::size_t size);

	void onReadAvail(const void *data, std::size_t size) = 0;

public:
	bool send(StreamBuffer buffer, bool fin = false) FINAL;
	bool hasBeenShutdown() const FINAL;
	bool forceShutdown() FINAL;

	boost::shared_ptr<const HttpSession> getParent() const {
		return m_parent.lock();
	}
	boost::shared_ptr<HttpSession> getParent(){
		return m_parent.lock();
	}

	boost::shared_ptr<const HttpSession> getSafeParent() const {
		return boost::shared_ptr<const HttpSession>(m_parent);
	}
	boost::shared_ptr<HttpSession> getSafeParent(){
		return boost::shared_ptr<HttpSession>(m_parent);
	}

	std::size_t getCategory() const;
	const std::string &getUri() const;
	const OptionalMap &getGetParams() const;
	const OptionalMap &getHeaders() const;
};

}

#endif
