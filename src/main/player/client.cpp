// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "client.hpp"
#include "exception.hpp"
#include "error_protocol.hpp"
#include "../singletons/timer_daemon.hpp"
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

void keepAliveTimerProc(boost::weak_ptr<PlayerClient> weak){
    const AUTO(client, weak.lock());
    if(!client){
    	return;
    }
    LOG_POSEIDON_TRACE("Sending heartbeat packet...");
    client->send(PlayerErrorProtocol::ID,
    	StreamBuffer(PlayerErrorProtocol(PlayerErrorProtocol::ID, 0, VAL_INIT)), false);
}

}

PlayerClient::PlayerClient(const IpPort &addr, unsigned long long keepAliveTimeout, bool useSsl)
	: TcpClientBase(addr, useSsl)
	, m_payloadLen((boost::uint64_t)-1), m_protocolId(0)
{
	if(keepAliveTimeout != 0){
		m_keepAliveTimer = TimerDaemon::registerTimer(keepAliveTimeout, keepAliveTimeout,
			boost::bind(&keepAliveTimerProc, virtualWeakFromThis<PlayerClient>()));
	}
}
PlayerClient::~PlayerClient(){
	if(m_payloadLen != (boost::uint64_t)-1){
		LOG_POSEIDON_WARN(
			"Now that this session is to be destroyed, a premature response has to be discarded.");
	}
}

void PlayerClient::onReadAvail(const void *data, std::size_t size){
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
			}
			if(m_payload.size() < m_payloadLen){
				break;
			}
			pendJob(boost::make_shared<ServerResponseJob>(virtualSharedFromThis<PlayerClient>(),
				m_protocolId, m_payload.cut(m_payloadLen)));
			m_payloadLen = (boost::uint64_t)-1;
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
