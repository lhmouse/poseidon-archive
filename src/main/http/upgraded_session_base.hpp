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
private:
	const boost::weak_ptr<HttpSession> m_parent;

protected:
	explicit HttpUpgradedSessionBase(boost::weak_ptr<HttpSession> parent);

public:
	virtual void onInitContents(const void *data, std::size_t size);
	const std::string &getRemoteIp() const;
	void onReadAvail(const void *data, std::size_t size) = 0;
	bool send(StreamBuffer buffer);
	bool hasBeenShutdown() const;
	bool shutdown();
	virtual bool shutdown(StreamBuffer buffer);
	bool forceShutdown();

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

	const std::string &getUri() const;
	const OptionalMap &getParams() const;
	const OptionalMap &getHeaders() const;
};

}

#endif
