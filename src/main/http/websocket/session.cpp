// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014, LH_Mouse. All wrongs reserved.

#include "../../precompiled.hpp"
#include "session.hpp"
#include "exception.hpp"
#include "../session.hpp"
#include "../../optional_map.hpp"
#include "../../singletons/job_dispatcher.hpp"
#include "../../singletons/websocket_servlet_depository.hpp"
#include "../../log.hpp"
#include "../../utilities.hpp"
#include "../../endian.hpp"
#include "../../job_base.hpp"
#include "../../profiler.hpp"
using namespace Poseidon;

namespace {

class WebSocketRequestJob : public JobBase {
private:
	const boost::shared_ptr<WebSocketSession> m_session;
	const std::string m_uri;
	const WebSocketOpCode m_opcode;

	StreamBuffer m_payload;

public:
	WebSocketRequestJob(boost::shared_ptr<WebSocketSession> session,
		std::string uri, WebSocketOpCode opcode, StreamBuffer payload)
		: m_session(STD_MOVE(session)), m_uri(STD_MOVE(uri)), m_opcode(opcode)
		, m_payload(STD_MOVE(payload))
	{
	}

protected:
	void perform(){
		PROFILE_ME;

		try {
			const AUTO(servlet, WebSocketServletDepository::getServlet(
				m_session->getCategory(), m_uri.c_str()));
			if(!servlet){
				LOG_POSEIDON_WARN("No servlet for URI ", m_uri);
				DEBUG_THROW(WebSocketException, WS_INACCEPTABLE, "Unknown URI");
				return;
			}

			LOG_POSEIDON_DEBUG("Dispatching packet: URI = ", m_uri,
				", payload size = ", m_payload.size());
			(*servlet)(m_session, m_opcode, STD_MOVE(m_payload));
		} catch(WebSocketException &e){
			LOG_POSEIDON_ERROR("WebSocketException thrown in websocket servlet, status = ", e.status(),
				", what = ", e.what());
			// m_session->shutdown(e.status(), StreamBuffer(e.what()));
			throw;
		} catch(...){
			LOG_POSEIDON_ERROR("Forwarding exception...");
			// m_session->shutdown(WS_INTERNAL_ERROR);
			throw;
		}
	}
};

}

WebSocketSession::WebSocketSession(const boost::shared_ptr<HttpSession> &parent)
	: HttpUpgradedSessionBase(parent)
	, m_state(ST_OPCODE)
{
}

void WebSocketSession::onInitContents(const void *data, std::size_t size){
	PROFILE_ME;

	(void)data;
	(void)size;

	setTimeout(WebSocketServletDepository::getKeepAliveTimeout());
}
void WebSocketSession::onReadAvail(const void *data, std::size_t size){
	PROFILE_ME;

	m_payload.put(data, size);
	try {
		for(;;){
			switch(m_state){
				int ch;
				boost::uint16_t temp16;
				boost::uint32_t temp32;
				boost::uint64_t temp64;
				std::size_t remaining;

			case ST_OPCODE:
				ch = m_payload.get();
				if(ch == -1){
					goto exit_for;
				}
				if(ch & (WS_FL_RSV1 | WS_FL_RSV2 | WS_FL_RSV3)){
					LOG_POSEIDON_WARN("Aborting because some reserved bits are set, opcode = ", ch);
					DEBUG_THROW(WebSocketException, WS_PROTOCOL_ERROR, "Reserved bits set");
				}
				m_fin = ch & WS_FL_FIN;
				m_opcode = static_cast<WebSocketOpCode>(ch & WS_FL_OPCODE);
				if((m_opcode & WS_FL_CONTROL) && !m_fin){
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
					m_payload.get(&temp16, 2);
					m_payloadLen = loadBe(temp16);
				} else {
					if(m_payload.size() < 8){
						goto exit_for;
					}
					m_payload.get(&temp64, 8);
					m_payloadLen = loadBe(temp64);
				}
				m_state = ST_MASK;
				break;

			case ST_MASK:
				LOG_POSEIDON_DEBUG("Payload length = ", m_payloadLen);

				if(m_payload.size() < 4){
					goto exit_for;
				}
				m_payload.get(&temp32, 4);
				m_payloadMask = loadLe(temp32);
				m_state = ST_PAYLOAD;
				break;

			case ST_PAYLOAD:
				remaining = m_payloadLen - m_whole.size();
				if(m_whole.size() + remaining >= WebSocketServletDepository::getMaxRequestLength()){
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
				if((m_opcode & WS_FL_CONTROL) != 0){
					onControlFrame();
				} else if(m_fin){
					pendJob(boost::make_shared<WebSocketRequestJob>(
						virtualSharedFromThis<WebSocketSession>(), getUri(), m_opcode, STD_MOVE(m_whole)));
					m_whole.clear();
				}
				setTimeout(WebSocketServletDepository::getKeepAliveTimeout());
				m_state = ST_OPCODE;
				break;
			}
		}
	exit_for:
		;
	} catch(WebSocketException &e){
		LOG_POSEIDON_ERROR("WebSocketException thrown while parseing data, status = ", e.status(),
			", what = ", e.what());
		shutdown(e.status(), StreamBuffer(e.what()));
		throw;
	} catch(...){
		LOG_POSEIDON_ERROR("Forwarding exception... shutdown the session first.");
		shutdown(WS_INTERNAL_ERROR);
		throw;
	}
}

void WebSocketSession::onControlFrame(){
	LOG_POSEIDON_DEBUG("Control frame, opcode = ", m_opcode);

	const AUTO(parent, getSafeParent());

	switch(m_opcode){
	case WS_CLOSE:
		LOG_POSEIDON_INFO("Received close frame from ", parent->getRemoteInfo());
		sendFrame(STD_MOVE(m_whole), WS_CLOSE, true, false);
		break;

	case WS_PING:
		LOG_POSEIDON_INFO("Received ping frame from ", parent->getRemoteInfo());
		sendFrame(STD_MOVE(m_whole), WS_PONG, false, false);
		break;

	case WS_PONG:
		LOG_POSEIDON_INFO("Received pong frame from ", parent->getRemoteInfo());
		break;

	default:
		DEBUG_THROW(WebSocketException, WS_PROTOCOL_ERROR, "Invalid opcode");
		break;
	}
}

bool WebSocketSession::sendFrame(StreamBuffer contents, WebSocketOpCode opcode, bool fin, bool masked){
	StreamBuffer packet;
	unsigned char ch = opcode | WS_FL_FIN;
	packet.put(ch);
	const std::size_t size = contents.size();
	ch = masked ? 0x80 : 0;
	if(size < 0x7E){
		ch |= size;
		packet.put(ch);
	} else if(size < 0x10000){
		ch |= 0x7E;
		packet.put(ch);
		boost::uint16_t temp;
		storeBe(temp, size);
		packet.put(&temp, 2);
	} else {
		ch |= 0x7F;
		packet.put(ch);
		boost::uint64_t temp;
		storeBe(temp, size);
		packet.put(&temp, 8);
	}
	if(masked){
		boost::uint32_t mask;
		storeLe(mask, rand32());
		packet.put(&mask, 4);
		int ch;
		for(;;){
			ch = contents.get();
			if(ch == -1){
				break;
			}
			ch ^= mask;
			packet.put(ch);
			mask = (mask << 24) | (mask >> 8);
		}
	} else {
		packet.splice(contents);
	}
	return HttpUpgradedSessionBase::send(STD_MOVE(contents), fin);
}

bool WebSocketSession::send(StreamBuffer contents, bool binary, bool fin, bool masked){
	if(!sendFrame(STD_MOVE(contents), binary ? WS_DATA_BIN : WS_DATA_TEXT, false, masked)){
		return false;
	}
	if(fin && !shutdown(WS_NORMAL_CLOSURE)){
		return false;
	}
	return true;
}
bool WebSocketSession::shutdown(WebSocketStatus status, StreamBuffer additional){
	StreamBuffer temp;
	boost::uint16_t codeBe;
	storeBe(codeBe, static_cast<unsigned>(status));
	temp.put(&codeBe, 2);
	temp.splice(additional);
	return sendFrame(STD_MOVE(temp), WS_CLOSE, true, false);
}
