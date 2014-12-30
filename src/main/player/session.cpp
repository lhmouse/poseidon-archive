// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "session.hpp"
#include "exception.hpp"
#include "error_protocol.hpp"
#include "../log.hpp"
#include "../exception.hpp"
#include "../singletons/player_servlet_depository.hpp"
#include "../job_base.hpp"
#include "../profiler.hpp"
#include "../endian.hpp"
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
				LOG_POSEIDON_DEBUG("Received error packet: status = ", packet.status,
					", reason = ", packet.reason);
				if(packet.status != PLAYER_OK){
					LOG_POSEIDON_DEBUG("Shutting down session as requested...");
					session->sendError(PlayerErrorProtocol::ID,
						static_cast<PlayerStatus>(packet.status), STD_MOVE(packet.reason), true);
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
		} catch(PlayerProtocolException &e){
			LOG_POSEIDON_ERROR("PlayerProtocolException thrown in player servlet, protocol id = ",
				m_protocolId, ", status = ", static_cast<unsigned>(e.status()), ", what = ", e.what());
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
				if(m_payload.size() < 4){
					break;
				}
				boost::uint16_t temp16;
				m_payload.peek(&temp16, 2);
				if((temp16 & 0x8000) != 0){
					if(m_payload.size() < 10){
						break;
					}
					boost::uint64_t temp64;
					m_payload.get(&temp64, 8);
					m_payloadLen = loadBe(temp64) & 0x7FFFFFFFFFFFFFFFull;
				} else {
					m_payload.discard(2);
					m_payloadLen = loadBe(temp16);
				}
				m_payload.get(&temp16, 2);
				m_protocolId = loadBe(temp16);
				LOG_POSEIDON_DEBUG("Protocol len = ", m_payloadLen, ", id = ", m_protocolId);

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
			setTimeout(PlayerServletDepository::getKeepAliveTimeout());
			m_payloadLen = (boost::uint64_t)-1;
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
	StreamBuffer data;
	const std::size_t size = contents.size();
	if(size < 0x8000){
		boost::uint16_t temp16;
		storeBe(temp16, size);
		data.put(&temp16, 2);
	} else {
		boost::uint64_t temp64;
		storeBe(temp64, size | 0x8000000000000000ull);
		data.put(&temp64, 8);
	}
	boost::uint16_t temp16;
	storeBe(temp16, protocolId);
	data.put(&temp16, 2);
	data.splice(contents);
	return TcpSessionBase::send(STD_MOVE(data), fin);
}

bool PlayerSession::sendError(boost::uint16_t protocolId, PlayerStatus status,
	std::string reason, bool fin)
{
	return send(PlayerErrorProtocol::ID, StreamBuffer(PlayerErrorProtocol(
		protocolId, static_cast<unsigned>(status), STD_MOVE(reason))), fin);
}
