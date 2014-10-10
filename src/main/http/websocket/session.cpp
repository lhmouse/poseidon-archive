#include "../../../precompiled.hpp"
#include "session.hpp"
#include "exception.hpp"
#include "../session.hpp"
#include "../../optional_map.hpp"
#include "../../singletons/job_dispatcher.hpp"
#include "../../log.hpp"
#include "../../utilities.hpp"
using namespace Poseidon;

namespace {

bool sendFrame(WebSocketSession *session,
	WebSocketOpCode opcode, StreamBuffer buffer, bool masked)
{
	StreamBuffer frame;
	unsigned char ch = 0x80 | opcode;
	frame.put(ch);
	const std::size_t size = buffer.size();
	ch = masked ? 0x80 : 0;
	if(size < 0xFE){
		ch |= size;
		frame.put(ch);
	} else if(size < 0x10000){
		ch |= 0xFE;
		frame.put(ch);
		const boost::uint16_t temp = htobe16(size);
		frame.put(&temp, 2);
	} else {
		ch |= 0xFF;
		frame.put(ch);
		const boost::uint64_t temp = htobe64(size);
		frame.put(&temp, 8);
	}
	if(masked){
		boost::uint32_t mask = htole32(rand32());
		frame.put(&mask, 4);
		int ch;
		for(;;){
			ch = buffer.get();
			if(ch == -1){
				break;
			}
			ch ^= mask;
			frame.put(ch);
			mask = (mask << 24) | (mask >> 8);
		}
	} else {
		frame.splice(buffer);
	}
	return session->HttpUpgradedSessionBase::send(STD_MOVE(frame));
}

}

WebSocketSession::WebSocketSession(boost::weak_ptr<HttpSession> parent)
	: HttpUpgradedSessionBase(STD_MOVE(parent))
	, m_state(ST_OPCODE)
{
}

void WebSocketSession::onReadAvail(const void *data, std::size_t size){
	m_incoming.put(data, size);
	try {
		DEBUG_THROW(WebSocketException, WS_PROTOCOL_ERROR);
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
	return sendFrame(this, binary ? WS_DATA_BIN : WS_DATA_TEXT, STD_MOVE(buffer), masked);
}
bool WebSocketSession::shutdown(WebSocketStatus status){
	StreamBuffer frame;
	const boost::uint16_t codeBe = htobe16(static_cast<unsigned>(status));
	frame.put(&codeBe, 2);
	frame.put(getWebSocketStatusDesc(status));
	return HttpUpgradedSessionBase::shutdown(STD_MOVE(frame));
}
bool WebSocketSession::shutdown(WebSocketStatus status, StreamBuffer reason){
	StreamBuffer frame;
	const boost::uint16_t codeBe = htobe16(static_cast<unsigned>(status));
	frame.put(&codeBe, 2);
	frame.splice(reason);
	return HttpUpgradedSessionBase::shutdown(STD_MOVE(frame));
}
