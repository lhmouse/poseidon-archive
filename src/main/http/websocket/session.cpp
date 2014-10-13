#include "../../../precompiled.hpp"
#include "session.hpp"
#include "exception.hpp"
#include "../session.hpp"
#include "../../optional_map.hpp"
#include "../../singletons/job_dispatcher.hpp"
#include "../../singletons/websocket_servlet_manager.hpp"
#include "../../log.hpp"
#include "../../utilities.hpp"
#include "../../job_base.hpp"
#include "../../profiler.hpp"
using namespace Poseidon;

namespace {

const std::size_t MAX_PACKET_SIZE = 0x4000;

StreamBuffer makeFrame(WebSocketOpCode opcode, StreamBuffer buffer, bool masked){
	StreamBuffer ret;
	unsigned char ch = opcode | WS_FL_FIN;
	ret.put(ch);
	const std::size_t size = buffer.size();
	ch = masked ? 0x80 : 0;
	if(size < 0x7E){
		ch |= size;
		ret.put(ch);
	} else if(size < 0x10000){
		ch |= 0x7E;
		ret.put(ch);
		const boost::uint16_t temp = htobe16(size);
		ret.put(&temp, 2);
	} else {
		ch |= 0x7F;
		ret.put(ch);
		const boost::uint64_t temp = htobe64(size);
		ret.put(&temp, 8);
	}
	if(masked){
		boost::uint32_t mask = htole32(rand32());
		ret.put(&mask, 4);
		int ch;
		for(;;){
			ch = buffer.get();
			if(ch == -1){
				break;
			}
			ch ^= mask;
			ret.put(ch);
			mask = (mask << 24) | (mask >> 8);
		}
	} else {
		ret.splice(buffer);
	}
	return ret;
}

class WebSocketRequestJob : public JobBase {
private:
	const std::string m_uri;
	const boost::weak_ptr<WebSocketSession> m_session;

	const WebSocketOpCode m_opcode;
	StreamBuffer m_payload;

public:
	WebSocketRequestJob(std::string uri, boost::weak_ptr<WebSocketSession> session,
		WebSocketOpCode opcode, StreamBuffer payload)
		: m_uri(STD_MOVE(uri)), m_session(STD_MOVE(session))
		, m_opcode(opcode), m_payload(STD_MOVE(payload))
	{
	}

protected:
	void perform(){
		AUTO(session, m_session.lock());
		if(!session){
			LOG_WARNING("The specified WebSocket session has expired.");
			return;
		}
		boost::shared_ptr<const void> lockedDep;
		const AUTO(servlet, WebSocketServletManager::getServlet(lockedDep, m_uri));
		if(!servlet){
			LOG_WARNING("No servlet for URI ", m_uri);
			session->shutdown(WS_INACCEPTABLE);
			return;
		}
		LOG_DEBUG("Dispatching packet: URI = ", m_uri, ", payload size = ", m_payload.size());
		(*servlet)(STD_MOVE(session), m_opcode, STD_MOVE(m_payload));
	}
};

}

WebSocketSession::WebSocketSession(boost::weak_ptr<HttpSession> parent)
	: HttpUpgradedSessionBase(STD_MOVE(parent))
	, m_state(ST_OPCODE)
{
}

void WebSocketSession::onControlFrame(){
	LOG_DEBUG("Control frame, opcode = ", m_opcode);

	switch(m_opcode){
	case WS_CLOSE:
		LOG_INFO("Received close frame from ", getRemoteIp(),
			", the connection will be shut down.");
		HttpUpgradedSessionBase::shutdown(makeFrame(WS_CLOSE, STD_MOVE(m_whole), false));
		break;

	case WS_PING:
		LOG_INFO("Received ping frame from ", getRemoteIp());
		HttpUpgradedSessionBase::send(makeFrame(WS_PONG, STD_MOVE(m_whole), false));
		break;

	case WS_PONG:
		LOG_INFO("Received pong frame from ", getRemoteIp());
		break;

	default:
		DEBUG_THROW(WebSocketException, WS_PROTOCOL_ERROR, "Invalid opcode");
		break;
	}
}

void WebSocketSession::onReadAvail(const void *data, std::size_t size){
	PROFILE_ME;

	m_payload.put(data, size);
	try {
		for(;;){
			int ch;
			std::size_t remaining;

			switch(m_state){
			case ST_OPCODE:
				ch = m_payload.get();
				if(ch == -1){
					goto exit_for;
				}
				if(ch & (WS_FL_RSV1 | WS_FL_RSV2 | WS_FL_RSV3)){
					LOG_WARNING("Aborting because some reserved bits are set, opcode = ", ch);
					DEBUG_THROW(WebSocketException, WS_PROTOCOL_ERROR, "Reserved bits set");
				}
				m_final = ch & WS_FL_FIN;
				m_opcode = static_cast<WebSocketOpCode>(ch & WS_FL_OPCODE);
				if((m_opcode & WS_FL_CONTROL) && !m_final){
					DEBUG_THROW(WebSocketException, WS_PROTOCOL_ERROR, "Control frame fragemented");
				}
				m_state = ST_PAYLOAD_LEN;
				break;

			case ST_PAYLOAD_LEN:
				ch = m_payload.get();
				if(ch == -1){
					goto exit_for;
				}
				if((ch & 0x80) == 0){
					DEBUG_THROW(WebSocketException, WS_ACCESS_DENIED, "Non-masked frames not allowed");
				}
				m_payloadLen = (unsigned char)(ch & 0x7F);
				if(m_payloadLen >= 0x7E){
					if(m_opcode & WS_FL_CONTROL){
						DEBUG_THROW(WebSocketException, WS_PROTOCOL_ERROR, "Control frame too large");
					}
					m_state = ST_EX_PAYLOAD_LEN;
				} else {
					m_state = ST_MASK;
				}
				break;

			case ST_EX_PAYLOAD_LEN:
				if(m_payloadLen == 0x7E){
					if(m_payload.size() < 2){
						goto exit_for;
					}
					boost::uint16_t temp;
					m_payload.get(&temp, 2);
					m_payloadLen = be16toh(temp);
				} else {
					if(m_payload.size() < 8){
						goto exit_for;
					}
					m_payload.get(&m_payloadLen, 8);
					m_payloadLen = be64toh(m_payloadLen);
				}
				m_state = ST_MASK;
				break;

			case ST_MASK:
				LOG_DEBUG("Payload length = ", m_payloadLen);

				if(m_payload.size() < 4){
					goto exit_for;
				}
				m_payload.get(&m_payloadMask, 4);
				m_payloadMask = le32toh(m_payloadMask);
				m_state = ST_PAYLOAD;
				break;

			case ST_PAYLOAD:
				remaining = m_payloadLen - m_whole.size();
				if(m_whole.size() + remaining >= MAX_PACKET_SIZE){
					DEBUG_THROW(WebSocketException, WS_MESSAGE_TOO_LARGE, "Packet too large");
				}
				if(m_payload.size() < remaining){
					goto exit_for;
				}
				for(std::size_t i = 0; i < remaining; ++i){
					ch = m_payload.get();
					ch ^= m_payloadMask;
					m_payloadMask = (m_payloadMask << 24) | (m_payloadMask >> 8);
					m_whole.put(ch);
				}
				if(m_final){
					if((m_opcode & WS_FL_CONTROL) == 0){
						boost::make_shared<WebSocketRequestJob>(boost::ref(getUri()),
							virtualWeakFromThis<WebSocketSession>(), m_opcode, STD_MOVE(m_whole)
							)->pend();
					} else {
						onControlFrame();
					}
					m_whole.clear();
				}
				m_state = ST_OPCODE;
				break;
			}
		}
	exit_for:
		;
	} catch(WebSocketException &e){
		LOG_ERROR("WebSocketException thrown while reading, status = ", e.status(),
			", file = ", e.file(), ", line = ", e.line(), ", what = ", e.what());
		shutdown(e.status(), e.what());
		throw;
	} catch(...){
		LOG_ERROR("Forwarding exception... shutdown the session first.");
		shutdown(WS_INTERNAL_ERROR);
		throw;
	}
}
bool WebSocketSession::send(StreamBuffer buffer, bool binary, bool masked){
	return HttpUpgradedSessionBase::send(
		makeFrame(binary ? WS_DATA_BIN : WS_DATA_TEXT, STD_MOVE(buffer), masked));
}
bool WebSocketSession::shutdown(WebSocketStatus status){
	return shutdown(status, StreamBuffer(getWebSocketStatusDesc(status)));
}
bool WebSocketSession::shutdown(WebSocketStatus status, StreamBuffer reason){
	StreamBuffer payload;
	const boost::uint16_t codeBe = htobe16(static_cast<unsigned>(status));
	payload.put(&codeBe, 2);
	payload.splice(reason);
	return HttpUpgradedSessionBase::shutdown(makeFrame(WS_CLOSE, STD_MOVE(payload), false));
}
