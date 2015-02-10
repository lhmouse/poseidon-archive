// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "client.hpp"
#include "exception.hpp"
#include "error_message.hpp"
#include "../singletons/timer_daemon.hpp"
#include "../log.hpp"
#include "../exception.hpp"
#include "../job_base.hpp"
#include "../profiler.hpp"
using namespace Poseidon;

namespace {

class ServerResponseJob : public JobBase {
private:
	const boost::weak_ptr<PlayerClient> m_client;
	const unsigned m_messageId;

	StreamBuffer m_payload;

public:
	ServerResponseJob(boost::weak_ptr<PlayerClient> client,
		unsigned messageId, StreamBuffer payload)
		: m_client(STD_MOVE(client)), m_messageId(messageId)
		, m_payload(STD_MOVE(payload))
	{
	}

protected:
	void perform(){
		PROFILE_ME;

		const boost::shared_ptr<PlayerClient> client(m_client);
		try {
			if(m_messageId != PlayerErrorMessage::ID){
				LOG_POSEIDON_DEBUG("Dispatching: message = ", m_messageId, ", payload size = ", m_payload.size());
				client->onResponse(m_messageId, STD_MOVE(m_payload));
			} else {
				PlayerErrorMessage error(m_payload);
				LOG_POSEIDON_DEBUG("Dispatching error message: message id = ", error.messageId,
					", status = ", error.status, ", reason = ", error.reason);
				client->onError(error.messageId, PlayerStatus(error.status), STD_MOVE(error.reason));
			}
		} catch(PlayerMessageException &e){
			LOG_POSEIDON_ERROR("PlayerMessageException thrown in player servlet, message id = ", m_messageId,
				", status = ", static_cast<int>(e.status()), ", what = ", e.what());
			throw;
		} catch(...){
			LOG_POSEIDON_ERROR("Forwarding exception... message id = ", m_messageId);
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
    client->send(PlayerErrorMessage::ID, StreamBuffer(PlayerErrorMessage(PlayerErrorMessage::ID, 0, VAL_INIT)), false);
}

}

PlayerClient::PlayerClient(const IpPort &addr, boost::uint64_t keepAliveTimeout, bool useSsl)
	: TcpClientBase(addr, useSsl)
	, m_keepAliveTimeout(keepAliveTimeout)
	, m_payloadLen((boost::uint64_t)-1), m_messageId(0)
{
}
PlayerClient::~PlayerClient(){
	if(m_payloadLen != (boost::uint64_t)-1){
		LOG_POSEIDON_WARN("Now that this session is to be destroyed, a premature response has to be discarded.");
	}
}

void PlayerClient::onReadAvail(const void *data, std::size_t size){
	PROFILE_ME;

	try {
		if((m_keepAliveTimeout != 0) && !m_keepAliveTimer){
			m_keepAliveTimer = TimerDaemon::registerTimer(m_keepAliveTimeout, m_keepAliveTimeout,
				boost::bind(&keepAliveTimerProc, virtualWeakFromThis<PlayerClient>()));
		}

		m_payload.put(data, size);
		for(;;){
			if(m_payloadLen == (boost::uint64_t)-1){
				boost::uint16_t messageId;
				boost::uint64_t payloadLen;
				if(!PlayerMessageBase::decodeHeader(messageId, payloadLen, m_payload)){
					break;
				}
				m_messageId = messageId;
				m_payloadLen = payloadLen;
				LOG_POSEIDON_DEBUG("Message id = ", m_messageId, ", len = ", m_payloadLen);
			}
			if(m_payload.size() < m_payloadLen){
				break;
			}
			pendJob(boost::make_shared<ServerResponseJob>(virtualWeakFromThis<PlayerClient>(),
				m_messageId, m_payload.cut(m_payloadLen)));
			m_payloadLen = (boost::uint64_t)-1;
			m_messageId = 0;
		}
	} catch(PlayerMessageException &e){
		LOG_POSEIDON_ERROR(
			"PlayerMessageException thrown while parsing data, message id = ", m_messageId,
			", status = ", static_cast<int>(e.status()), ", what = ", e.what());
		throw;
	} catch(...){
		LOG_POSEIDON_ERROR("Forwarding exception... message id = ", m_messageId);
		throw;
	}
}

bool PlayerClient::send(boost::uint16_t messageId, StreamBuffer contents, bool fin){
	StreamBuffer data;
	PlayerMessageBase::encodeHeader(data, messageId, contents.size());
	data.splice(contents);
	return TcpSessionBase::send(STD_MOVE(data), fin);
}
