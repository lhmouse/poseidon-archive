// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "session.hpp"
#include "exception.hpp"
#include "control_code.hpp"
#include "error_protocol.hpp"
#include "../log.hpp"
#include "../exception.hpp"
#include "../singletons/player_servlet_depository.hpp"
#include "../job_base.hpp"
#include "../profiler.hpp"
using namespace Poseidon;

namespace {

class PlayerRequestJob : public JobBase {
private:
	const boost::weak_ptr<PlayerSession> m_session;
	const unsigned m_protocolId;

	StreamBuffer m_payload;

public:
	PlayerRequestJob(boost::weak_ptr<PlayerSession> session,
		unsigned protocolId, StreamBuffer payload)
		: m_session(STD_MOVE(session)), m_protocolId(protocolId)
		, m_payload(STD_MOVE(payload))
	{
	}

protected:
	void perform(){
		PROFILE_ME;

		const boost::shared_ptr<PlayerSession> session(m_session);
		try {
			if(m_protocolId == PlayerErrorProtocol::ID){
				PlayerErrorProtocol packet(m_payload);
				LOG_POSEIDON_DEBUG("Received error packet: protocol id = ", packet.protocolId,
					", status = ", packet.status, ", reason = ", packet.reason);

				switch(static_cast<PlayerControlCode>(packet.protocolId)){
				case PLAYER_CTL_HEARTBEAT:
					LOG_POSEIDON_TRACE("Received heartbeat from ", session->getRemoteInfo());
					break;

				default:
					LOG_POSEIDON_WARN("Unknown control code: ", packet.protocolId);
					session->send(PlayerErrorProtocol::ID, StreamBuffer(packet), true);
					break;
				}
			} else {
				const AUTO(category, session->getCategory());
				const AUTO(servlet, PlayerServletDepository::getServlet(category, m_protocolId));
				if(!servlet){
					LOG_POSEIDON_WARN(
						"No servlet in category ", category, " matches protocol ", m_protocolId);
					DEBUG_THROW(PlayerProtocolException, PLAYER_NOT_FOUND,
						SharedNts::observe("Unknown protocol"));
				}

				LOG_POSEIDON_DEBUG("Dispatching packet: protocol = ", m_protocolId,
					", payload size = ", m_payload.size());
				(*servlet)(session, STD_MOVE(m_payload));
			}
			session->setTimeout(PlayerServletDepository::getKeepAliveTimeout());
		} catch(PlayerProtocolException &e){
			LOG_POSEIDON_ERROR("PlayerProtocolException thrown in player servlet, protocol id = ",
				m_protocolId, ", status = ", static_cast<int>(e.status()), ", what = ", e.what());
			session->sendError(m_protocolId, e.status(), e.what(), false);
			throw;
		} catch(...){
			LOG_POSEIDON_ERROR("Forwarding exception... protocol id = ", m_protocolId);
			session->sendError(m_protocolId, PLAYER_INTERNAL_ERROR, false);
			throw;
		}
	}
};

}

PlayerSession::PlayerSession(std::size_t category, UniqueFile socket)
	: TcpSessionBase(STD_MOVE(socket))
	, m_category(category)
	, m_payloadLen((boost::uint64_t)-1), m_protocolId(0)
{
}
PlayerSession::~PlayerSession(){
	if(m_payloadLen != (boost::uint64_t)-1){
		LOG_POSEIDON_WARN(
			"Now that this session is to be destroyed, a premature request has to be discarded.");
	}
}

void PlayerSession::onReadAvail(const void *data, std::size_t size){
	PROFILE_ME;

	try {
		m_payload.put(data, size);
		for(;;){
			if(m_payloadLen == (boost::uint64_t)-1){
				boost::uint16_t protocolId;
				boost::uint64_t payloadLen;
				if(!PlayerProtocolBase::decodeHeader(protocolId, payloadLen, m_payload)){
					break;
				}
				m_protocolId = protocolId;
				m_payloadLen = payloadLen;
				LOG_POSEIDON_DEBUG("Protocol id = ", m_protocolId, ", len = ", m_payloadLen);

				const std::size_t maxRequestLength = PlayerServletDepository::getMaxRequestLength();
				if((unsigned)m_payloadLen >= maxRequestLength){
					LOG_POSEIDON_WARN(
						"Request too large: size = ", m_payloadLen, ", max = ", maxRequestLength);
					DEBUG_THROW(PlayerProtocolException, PLAYER_REQUEST_TOO_LARGE,
						SharedNts::observe("Request too large"));
				}
			}
			if(m_payload.size() < (unsigned)m_payloadLen){
				break;
			}
			pendJob(boost::make_shared<PlayerRequestJob>(virtualWeakFromThis<PlayerSession>(),
				m_protocolId, m_payload.cut(m_payloadLen)));
			m_payloadLen = (boost::uint64_t)-1;
			m_protocolId = 0;
		}
	} catch(PlayerProtocolException &e){
		LOG_POSEIDON_ERROR(
			"PlayerProtocolException thrown while parsing data, protocol id = ", m_protocolId,
			", status = ", static_cast<int>(e.status()), ", what = ", e.what());
		sendError(m_protocolId, e.status(), e.what(), true);
		throw;
	} catch(...){
		LOG_POSEIDON_ERROR("Forwarding exception... protocol id = ", m_protocolId);
		sendError(m_protocolId, PLAYER_INTERNAL_ERROR, true);
		throw;
	}
}

bool PlayerSession::send(boost::uint16_t protocolId, StreamBuffer contents, bool fin){
	StreamBuffer data;
	PlayerProtocolBase::encodeHeader(data, protocolId, contents.size());
	data.splice(contents);
	return TcpSessionBase::send(STD_MOVE(data), fin);
}

bool PlayerSession::sendError(boost::uint16_t protocolId, PlayerStatus status,
	std::string reason, bool fin)
{
	return send(PlayerErrorProtocol::ID, StreamBuffer(PlayerErrorProtocol(
		protocolId, static_cast<int>(status), STD_MOVE(reason))), fin);
}
