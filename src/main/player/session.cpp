#include "../../precompiled.hpp"
#include "session.hpp"
#include "status.hpp"
#include "exception.hpp"
#include "protocol_base.hpp"
#include "../log.hpp"
#include "../exception.hpp"
#include "../singletons/player_servlet_manager.hpp"
#include "../job_base.hpp"
#include "../profiler.hpp"
using namespace Poseidon;

namespace {

#define PROTOCOL_NAME	ErrorResponse
#define PROTOCOL_FIELDS	\
	FIELD_VUINT(status)	\
	FIELD_VUINT(protocolId)
#include "protocol_generator.hpp"

}

namespace {

class PlayerRequestJob : public JobBase {
private:
	const unsigned m_protocolId;
	const boost::shared_ptr<PlayerSession> m_session;

	StreamBuffer m_payload;

public:
	PlayerRequestJob(unsigned protocolId,
		boost::shared_ptr<PlayerSession> session, StreamBuffer payload)
		: m_protocolId(protocolId), m_session(STD_MOVE(session))
		, m_payload(STD_MOVE(payload))
	{
	}

protected:
	void perform(){
		PROFILE_ME;

		try {
			boost::shared_ptr<const void> lockedDep;
			const AUTO(servlet, PlayerServletManager::getServlet(lockedDep, m_protocolId));
			if(!servlet){
				LOG_WARNING("No servlet for protocol ", m_protocolId);
				DEBUG_THROW(PlayerException, PLAYER_NOT_FOUND);
			}

			LOG_DEBUG("Dispatching: protocol = ", m_protocolId, ", payload size = ", m_payload.size());
			(*servlet)(m_session, STD_MOVE(m_payload));
		} catch(PlayerException &e){
			LOG_ERROR("PlayerException thrown in player servlet, protocol id = ", m_protocolId,
				", status = ", static_cast<unsigned>(e.status()));
			m_session->send(0, ErrorResponse(static_cast<unsigned>(e.status()), m_protocolId));
			throw;
		} catch(...){
			LOG_ERROR("Forwarding exception... protocol id = ", m_protocolId);
			m_session->send(0, ErrorResponse(static_cast<unsigned>(PLAYER_INTERNAL_ERROR), m_protocolId));
			throw;
		}
	}
};

}

PlayerSession::PlayerSession(Move<ScopedFile> socket)
	: TcpSessionBase(STD_MOVE(socket))
	, m_payloadLen(-1), m_protocolId(0)
{
}
PlayerSession::~PlayerSession(){
	if(m_payloadLen != -1){
		LOG_WARNING("Now that this player session is to be destroyed, "
			"a premature packet has to be discarded: payload size = ",
			m_payloadLen, ", read = ", m_payload.size());
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

				m_payloadLen = m_payload.get() & 0xFF;
				m_payloadLen |= (m_payload.get() & 0xFF) << 8;
				m_payloadLen &= 0x3FFF;

				m_protocolId = m_payload.get() & 0xFF;
				m_protocolId |= (m_payload.get() & 0xFF) << 8;

				LOG_DEBUG("Protocol len = ", m_payloadLen, ", id = ", m_protocolId);

				// 仅测试。
				boost::shared_ptr<const void> lockedDep;
				if(!PlayerServletManager::getServlet(lockedDep, m_protocolId)){
					DEBUG_THROW(PlayerException, PLAYER_NOT_FOUND);
				}
			}
			if(m_payload.size() < (unsigned)m_payloadLen){
				break;
			}
			boost::make_shared<PlayerRequestJob>(m_protocolId,
				virtualSharedFromThis<PlayerSession>(), m_payload.cut(m_payloadLen))->pend();
			m_protocolId = 0;
			m_payloadLen = -1;
		}
	} catch(PlayerException &e){
		LOG_ERROR("PlayerException while dispatching player data, protocol id = ", m_protocolId,
			", status = ", static_cast<unsigned>(e.status()));
		shutdown(e.status());
		throw;
	} catch(...){
		LOG_ERROR("Forwarding exception... protocol id = ", m_protocolId);
		shutdown(PLAYER_INTERNAL_ERROR);
		throw;
	}
}

bool PlayerSession::shutdown(PlayerStatus status,
	boost::uint16_t protocolId, StreamBuffer additional)
{
	ErrorResponse response(static_cast<unsigned>(status), protocolId);
	StreamBuffer protocolData;
	response >> protocolData;
	protocolData.splice(additional);

	StreamBuffer temp;
	const std::size_t size = protocolData.size();
	temp.put(size & 0xFF);
	temp.put(size >> 8);
	temp.put(protocolId & 0xFF);
	temp.put(protocolId >> 8);
	temp.splice(protocolData);
	return TcpSessionBase::shutdown(STD_MOVE(temp));
}

bool PlayerSession::send(boost::uint16_t protocolId, StreamBuffer protocolData){
	const std::size_t size = protocolData.size();
	if(size > 0xFFFF){
		LOG_WARNING("Respond packet too large, size = ", size);
		DEBUG_THROW(PlayerException, PLAYER_REQUEST_TOO_LARGE);
	}
	StreamBuffer temp;
	temp.put(size & 0xFF);
	temp.put(size >> 8);
	temp.put(protocolId & 0xFF);
	temp.put(protocolId >> 8);
	temp.splice(protocolData);
	return TcpSessionBase::send(STD_MOVE(temp));
}
