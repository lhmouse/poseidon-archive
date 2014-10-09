#ifndef POSEIDON_HTTP_WEBSOCKET_SESSION_HPP_
#define POSEIDON_HTTP_WEBSOCKET_SESSION_HPP_

#include "../upgraded_session_base.hpp"

namespace Poseidon {

class WebSocketSession : public HttpUpgradedSessionBase {
public:
	explicit WebSocketSession(boost::weak_ptr<HttpSession> parent);

protected:
	void onReadAvail(const void *data, std::size_t size) = 0;
	void sendUsingMove(StreamBuffer &buffer);
	void shutdown();
};

}

#endif
