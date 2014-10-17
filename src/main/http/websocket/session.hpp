#ifndef POSEIDON_HTTP_WEBSOCKET_SESSION_HPP_
#define POSEIDON_HTTP_WEBSOCKET_SESSION_HPP_

#include "../../../cxx_ver.hpp"
#include "../upgraded_session_base.hpp"
#include <boost/shared_ptr.hpp>
#include <boost/weak_ptr.hpp>
#include <boost/cstdint.hpp>
#include "../../stream_buffer.hpp"
#include "opcode.hpp"
#include "status.hpp"

namespace Poseidon {

class OptionalMap;
class HttpSession;

class WebSocketSession : public HttpUpgradedSessionBase {
private:
	enum State {
		ST_OPCODE,
		ST_PAYLOAD_LEN,
		ST_EX_PAYLOAD_LEN,
		ST_MASK,
		ST_PAYLOAD,
	};

private:
	State m_state;
	bool m_final;
	WebSocketOpCode m_opcode;
	boost::uint64_t m_payloadLen;
	boost::uint32_t m_payloadMask;
	StreamBuffer m_payload;
	StreamBuffer m_whole;

public:
	explicit WebSocketSession(boost::weak_ptr<HttpSession> parent);

private:
	void onReadAvail(const void *data, std::size_t size);

	void onControlFrame();

public:
	bool send(StreamBuffer buffer, bool binary = true, bool masked = false);
	bool shutdown(WebSocketStatus status);
	bool shutdown(WebSocketStatus status, StreamBuffer reason);
};

}

#endif
