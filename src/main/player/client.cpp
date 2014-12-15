// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "client.hpp"
#include "exception.hpp"
#include "error_protocol.hpp"
#include "../log.hpp"
#include "../exception.hpp"
#include "../job_base.hpp"
#include "../profiler.hpp"
#include "../endian.hpp"
using namespace Poseidon;

namespace {

class ServerResponseJob : public JobBase {
private:
	const boost::shared_ptr<PlayerClient> m_client;
	const unsigned m_protocolId;

	StreamBuffer m_payload;

public:
	ServerResponseJob(boost::shared_ptr<PlayerClient> client,
		unsigned protocolId, StreamBuffer payload)
		: m_client(STD_MOVE(client)), m_protocolId(protocolId)
		, m_payload(STD_MOVE(payload))
	{
	}

protected:
	void perform(){
		PROFILE_ME;

		try {
			if(m_protocolId != PlayerErrorProtocol::ID){
				LOG_POSEIDON_DEBUG("Dispatching: protocol = ", m_protocolId, ", payload size = ", m_payload.size());
				m_client->onResponse(m_protocolId, STD_MOVE(m_payload));
			} else {
				PlayerErrorProtocol error(m_payload);
				LOG_POSEIDON_DEBUG("Dispatching error protocol: protocol id = ", error.protocolId,
					", status = ", error.status, ", reason = ", error.reason);
				m_client->onError(error.protocolId, static_cast<PlayerStatus>(error.status), STD_MOVE(error.reason));
			}
		} catch(PlayerProtocolException &e){
			LOG_POSEIDON_ERROR("PlayerProtocolException thrown in player servlet, protocol id = ", m_protocolId,
				", status = ", static_cast<unsigned>(e.status()), ", what = ", e.what());
			throw;
		} catch(...){
			LOG_POSEIDON_ERROR("Forwarding exception... protocol id = ", m_protocolId);
			throw;
		}
	}
};

}

PlayerClient::PlayerClient(const IpPort &addr, bool useSsl)
	: TcpClientBase(addr, useSsl)
	, m_payloadLen(-1), m_protocolId(0)
{
}
PlayerClient::~PlayerClient(){
}

void PlayerClient::onReadAvail(const void *data, std::size_t size){
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
			}
			if(m_payload.size() < (unsigned)m_payloadLen){
				break;
			}
			pendJob(boost::make_shared<ServerResponseJob>(virtualSharedFromThis<PlayerClient>(),
				m_protocolId, m_payload.cut(m_payloadLen)));
			m_payloadLen = -1;
			m_protocolId = 0;
		}
	} catch(PlayerProtocolException &e){
		LOG_POSEIDON_ERROR(
			"PlayerProtocolException thrown while parsing data, protocol id = ", m_protocolId,
			", status = ", static_cast<unsigned>(e.status()), ", what = ", e.what());
		throw;
	} catch(...){
		LOG_POSEIDON_ERROR("Forwarding exception... protocol id = ", m_protocolId);
		throw;
	}
}

bool PlayerClient::send(boost::uint16_t protocolId, StreamBuffer contents, bool fin){
	const std::size_t size = contents.size();
	if(size > 0xFFFF){
		LOG_POSEIDON_WARN("Request packet too large, size = ", size);
		DEBUG_THROW(PlayerProtocolException, PLAYER_REQUEST_TOO_LARGE, "Request packet too large");
	}
	StreamBuffer data;
	boost::uint16_t tmp;
	storeLe(tmp, size);
	data.put(&tmp, 2);
	storeLe(tmp, protocolId);
	data.put(&tmp, 2);
	data.splice(contents);
	return TcpSessionBase::send(STD_MOVE(data), fin);
}
