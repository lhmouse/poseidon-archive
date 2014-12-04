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

const std::size_t MAX_PACKET_SIZE = 0x3FFF;

StreamBuffer makeFrame(WebSocketOpCode opcode, StreamBuffer contents, bool masked){
	StreamBuffer ret;
	unsigned char ch = opcode | WS_FL_FIN;
	ret.put(ch);
	const std::size_t size = contents.size();
	ch = masked ? 0x80 : 0;
	if(size < 0x7E){
		ch |= size;
		ret.put(ch);
	} else if(size < 0x10000){
		ch |= 0x7E;
		ret.put(ch);
		boost::uint16_t temp;
		storeBe(temp, size);
		ret.put(&temp, 2);
	} else {
		ch |= 0x7F;
		ret.put(ch);
		boost::uint64_t temp;
		storeBe(temp, size);
		ret.put(&temp, 8);
	}
	if(masked){
		boost::uint32_t mask;
		storeLe(mask, rand32());
		ret.put(&mask, 4);
		int ch;
		for(;;){
			ch = contents.get();
			if(ch == -1){
				break;
			}
			ch ^= mask;
			ret.put(ch);
			mask = (mask << 24) | (mask >> 8);
		}
	} else {
		ret.splice(contents);
	}
	return ret;
}

StreamBuffer makeCloseFrame(WebSocketStatus status, StreamBuffer contents){
	StreamBuffer temp;
	const boost::uint16_t codeBe = htobe16(static_cast<unsigned>(status));
	temp.put(&codeBe, 2);
	temp.splice(contents);
	return makeFrame(WS_CLOSE, STD_MOVE(temp), false);
}

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

void WebSocketSession::onReadAvail(const void *data, std::size_t size){
	PROFILE_ME;

	m_payload.put(data, size);
	try {
		for(;;){
			switch(m_state){
				int ch;

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
					boost::uint16_t temp;
					m_payload.get(&temp, 2);
					m_payloadLen = loadBe(temp);
				} else {
					if(m_payload.size() < 8){
						goto exit_for;
					}
					boost::uint64_t temp;
					m_payload.get(&temp, 8);
					m_payloadLen = loadBe(temp);
				}
				m_state = ST_MASK;
				break;

			case ST_MASK:
				LOG_POSEIDON_DEBUG("Payload length = ", m_payloadLen);

				if(m_payload.size() < 4){
					goto exit_for;
				}
				boost::uint32_t temp;
				m_payload.get(&temp, 4);
				m_payloadMask = loadLe(temp);
				m_state = ST_PAYLOAD;
				break;

			case ST_PAYLOAD:
				std::size_t remaining;
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
				if((m_opcode & WS_FL_CONTROL) != 0){
					onControlFrame();
				} else if(m_fin){
					boost::make_shared<WebSocketRequestJob>(
						virtualSharedFromThis<WebSocketSession>(),
						getUri(), m_opcode, STD_MOVE(m_whole))->pend();
					m_whole.clear();
				}
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
		HttpUpgradedSessionBase::send(makeFrame(WS_CLOSE, STD_MOVE(m_whole), false), true);
		break;

	case WS_PING:
		LOG_POSEIDON_INFO("Received ping frame from ", parent->getRemoteInfo());
		HttpUpgradedSessionBase::send(makeFrame(WS_PONG, STD_MOVE(m_whole), false), false);
		break;

	case WS_PONG:
		LOG_POSEIDON_INFO("Received pong frame from ", parent->getRemoteInfo());
		break;

	default:
		DEBUG_THROW(WebSocketException, WS_PROTOCOL_ERROR, "Invalid opcode");
		break;
	}
}

bool WebSocketSession::send(StreamBuffer contents, bool binary, bool fin, bool masked){
	if(!HttpUpgradedSessionBase::send(
		makeFrame(binary ? WS_DATA_BIN : WS_DATA_TEXT, STD_MOVE(contents), masked), false))
	{
		return false;
	}
	if(fin){
		return HttpUpgradedSessionBase::send(makeCloseFrame(WS_NORMAL_CLOSURE, StreamBuffer()), true);
	}
	return true;
}

bool WebSocketSession::shutdown(WebSocketStatus status, StreamBuffer additional){
	return HttpUpgradedSessionBase::send(makeCloseFrame(status, STD_MOVE(additional)), true);
}
