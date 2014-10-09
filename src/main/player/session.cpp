#include "../../precompiled.hpp"
#include "session.hpp"
#include "../log.hpp"
#include "../exception.hpp"
#include "../singletons/player_servlet_manager.hpp"
#include "../job_base.hpp"
#include "../profiler.hpp"
using namespace Poseidon;

namespace {

class PlayerRequestJob : public JobBase {
private:
	unsigned m_protocolId;
	boost::weak_ptr<PlayerSession> m_session;
	StreamBuffer m_packet;

	boost::shared_ptr<const void> m_lockedDep;

public:
	PlayerRequestJob(unsigned protocolId,
		boost::weak_ptr<PlayerSession> session, StreamBuffer packet)
		: m_protocolId(protocolId)
		, m_session(STD_MOVE(session)), m_packet(STD_MOVE(packet))
	{
	}

protected:
	void perform(){
		AUTO(session, m_session.lock());
		if(!session){
			LOG_WARNING("The specified player session has expired.");
			return;
		}

		const AUTO(servlet, PlayerServletManager::getServlet(m_lockedDep, m_protocolId));
		if(!servlet){
			LOG_WARNING("No servlet for protocol ", m_protocolId);
			session->shutdown();
			return;
		}
		LOG_DEBUG("Dispatching packet: protocol = ", m_protocolId,
			", payload size = ", m_packet.size());
		(*servlet)(STD_MOVE(session), STD_MOVE(m_packet));
	}
};

}

PlayerSession::PlayerSession(Move<ScopedFile> socket)
	: TcpSessionBase(STD_MOVE(socket))
	, m_payloadLen(-1), m_protocolId(0), m_payload()
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

	m_payload.put(data, size);
	for(;;){
		if(m_payloadLen == -1){
			if(m_payload.size() < 4){
				break;
			}
			m_protocolId = m_payload.get() & 0xFF;
			m_protocolId |= (m_payload.get() & 0xFF) << 8;

			m_payloadLen = m_payload.get() & 0xFF;
			m_payloadLen |= (m_payload.get() & 0xFF) << 8;
			m_payloadLen &= 0x3FFF;
		}
		if(m_payload.size() < (unsigned)m_payloadLen){
			break;
		}
		StreamBuffer packet = m_payload.cut(m_payloadLen);
		boost::make_shared<PlayerRequestJob>(m_protocolId,
			virtualWeakFromThis<PlayerSession>(), STD_MOVE(packet))->pend();
		m_payloadLen = -1;
	}
}

void PlayerSession::sendUsingMove(boost::uint16_t status, StreamBuffer &payload){
	const std::size_t size = payload.size();
	if(size > 0xFFFF){
		LOG_WARNING("Respond packet too large, size = ", size);
		DEBUG_THROW(Exception, "Packet too large: " + boost::lexical_cast<std::string>(size));
	}
	StreamBuffer temp;
	temp.put(status & 0xFF);
	temp.put(status >> 8);
	temp.put(size & 0xFF);
	temp.put(size >> 8);
	temp.splice(payload);
	TcpSessionBase::send(temp);
}
