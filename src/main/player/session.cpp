// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "session.hpp"
#include "status.hpp"
#include "exception.hpp"
#include "protocol_base.hpp"
#include "../log.hpp"
#include "../exception.hpp"
#include "../singletons/player_servlet_depository.hpp"
#include "../job_base.hpp"
#include "../profiler.hpp"
#include "../endian.hpp"
using namespace Poseidon;

namespace {

#define PROTOCOL_NAME	ErrorProtocol
#define PROTOCOL_ID		0
#define PROTOCOL_FIELDS	\
	FIELD_VUINT(protocolId)	\
	FIELD_VUINT(status)	\
	FIELD_STRING(reason)
#include "protocol_generator.hpp"

StreamBuffer makeResponse(boost::uint16_t protocolId, StreamBuffer contents){
	const std::size_t size = contents.size();
	if(size > 0xFFFF){
		LOG_POSEIDON_WARN("Respond packet too large, size = ", size);
		DEBUG_THROW(PlayerProtocolException, PLAYER_RESPOND_TOO_LARGE, "Respond packet too large");
	}
	StreamBuffer ret;
	boost::uint16_t tmp;
	storeLe(tmp, size);
	ret.put(&tmp, 2);
	storeLe(tmp, protocolId);
	ret.put(&tmp, 2);
	ret.splice(contents);
	return ret;
}

StreamBuffer makeErrorResponse(boost::uint16_t protocolId, PlayerStatus status, std::string reason){
	StreamBuffer temp;
	ErrorProtocol(protocolId, static_cast<unsigned>(status), STD_MOVE(reason)) >>temp;
	return makeResponse(ErrorProtocol::ID, STD_MOVE(temp));
}

class PlayerRequestJob : public JobBase {
private:
	const boost::shared_ptr<PlayerSession> m_session;
	const unsigned m_protocolId;

	StreamBuffer m_payload;

public:
	PlayerRequestJob(boost::shared_ptr<PlayerSession> session,
		unsigned protocolId, StreamBuffer payload)
		: m_session(STD_MOVE(session)), m_protocolId(protocolId)
		, m_payload(STD_MOVE(payload))
	{
	}

protected:
	void perform(){
		PROFILE_ME;

		try {
			const AUTO(servlet, PlayerServletDepository::getServlet(m_session->getCategory(), m_protocolId));
			if(!servlet){
				LOG_POSEIDON_WARN("No servlet for protocol ", m_protocolId);
				DEBUG_THROW(PlayerProtocolException, PLAYER_NOT_FOUND, "Unknown protocol");
			}

			LOG_POSEIDON_DEBUG("Dispatching: protocol = ", m_protocolId, ", payload size = ", m_payload.size());
			(*servlet)(m_session, STD_MOVE(m_payload));
		} catch(PlayerProtocolException &e){
			LOG_POSEIDON_ERROR("PlayerProtocolException thrown in player servlet, protocol id = ", m_protocolId,
				", status = ", static_cast<unsigned>(e.status()), ", what = ", e.what());
			m_session->sendError(m_protocolId, e.status(), e.what(), false);
			throw;
		} catch(...){
			LOG_POSEIDON_ERROR("Forwarding exception... protocol id = ", m_protocolId);
			m_session->sendError(m_protocolId, PLAYER_INTERNAL_ERROR, false);
			throw;
		}
	}
};

}

PlayerSession::PlayerSession(std::size_t category, UniqueFile socket)
	: TcpSessionBase(STD_MOVE(socket))
	, m_category(category)
	, m_payloadLen(-1), m_protocolId(0)
{
}
PlayerSession::~PlayerSession(){
	if(m_payloadLen != -1){
		LOG_POSEIDON_WARN(
			"Now that this session is to be destroyed, a premature request has to be discarded.");
	}
}

void PlayerSession::onReadAvail(const void *data, std::size_t size){
	PROFILE_ME;

	try {
		m_payload.put(data, size);
		for(;;){
			if(m_payloadLen == -1){
				if(m_payload.size() < 4){
					break;
				}
				boost::uint16_t tmp;
				m_payload.get(&tmp, 2);
				m_payloadLen = loadLe(tmp);
				m_payload.get(&tmp, 2);
				m_protocolId = loadLe(tmp);
				LOG_POSEIDON_DEBUG("Protocol len = ", m_payloadLen, ", id = ", m_protocolId);

				const std::size_t maxRequestLength = PlayerServletDepository::getMaxRequestLength();
				if((unsigned)m_payloadLen >= maxRequestLength){
					LOG_POSEIDON_WARN(
						"Request too large: size = ", m_payloadLen, ", max = ", maxRequestLength);
					DEBUG_THROW(PlayerProtocolException, PLAYER_REQUEST_TOO_LARGE, "Request too large");
				}
			}
			if(m_payload.size() < (unsigned)m_payloadLen){
				break;
			}
			boost::make_shared<PlayerRequestJob>(virtualSharedFromThis<PlayerSession>(),
				m_protocolId, m_payload.cut(m_payloadLen))->pend();
			m_payloadLen = -1;
			m_protocolId = 0;
		}
	} catch(PlayerProtocolException &e){
		LOG_POSEIDON_ERROR(
			"PlayerProtocolException thrown while parsing data, protocol id = ", m_protocolId,
			", status = ", static_cast<unsigned>(e.status()), ", what = ", e.what());
		sendError(m_protocolId, e.status(), e.what(), true);
		throw;
	} catch(...){
		LOG_POSEIDON_ERROR("Forwarding exception... protocol id = ", m_protocolId);
		sendError(m_protocolId, PLAYER_INTERNAL_ERROR, true);
		throw;
	}
}

bool PlayerSession::send(boost::uint16_t protocolId, StreamBuffer contents, bool fin){
	return TcpSessionBase::send(makeResponse(protocolId, STD_MOVE(contents)), fin);
}

bool PlayerSession::sendError(boost::uint16_t protocolId, PlayerStatus status,
	std::string reason, bool fin)
{
	return TcpSessionBase::send(makeErrorResponse(protocolId, status, STD_MOVE(reason)), fin);
}
