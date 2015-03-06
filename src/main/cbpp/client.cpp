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

namespace Poseidon {

namespace Cbpp {
	namespace {
		class ServerResponseJob : public JobBase {
		private:
			const boost::weak_ptr<Client> m_client;
			const unsigned m_messageId;
			const StreamBuffer m_payload;

		public:
			ServerResponseJob(boost::weak_ptr<Client> client, unsigned messageId, StreamBuffer payload)
				: m_client(STD_MOVE(client)), m_messageId(messageId), m_payload(STD_MOVE(payload))
			{
			}

		protected:
			boost::weak_ptr<const void> getCategory() const OVERRIDE {
				return m_client;
			}
			void perform() const OVERRIDE {
				PROFILE_ME;

				const boost::shared_ptr<Client> client(m_client);
				try {
					if(m_messageId != ErrorMessage::ID){
						LOG_POSEIDON_DEBUG("Dispatching: message = ", m_messageId, ", payload size = ", m_payload.size());
						client->onResponse(m_messageId, STD_MOVE(m_payload));
					} else {
						AUTO(payload, m_payload);
						ErrorMessage error(payload);
						LOG_POSEIDON_DEBUG("Dispatching error message: message id = ", error.messageId,
							", statusCode = ", error.statusCode, ", reason = ", error.reason);
						client->onError(error.messageId, StatusCode(error.statusCode), STD_MOVE(error.reason));
					}
				} catch(TryAgainLater &){
					throw;
				} catch(Exception &e){
					LOG_POSEIDON_ERROR("Cbpp::Exception thrown in  servlet, message id = ", m_messageId,
						", statusCode = ", e.statusCode(), ", what = ", e.what());
					throw;
				} catch(...){
					LOG_POSEIDON_ERROR("Forwarding exception... message id = ", m_messageId);
					client->shutdown(); // 关闭连接。
					throw;
				}
			}
		};

		void keepAliveTimerProc(const boost::weak_ptr<Client> &weak){
    		const AUTO(client, weak.lock());
    		if(!client){
    			return;
    		}
    		LOG_POSEIDON_TRACE("Sending heartbeat packet...");
    		client->send(ErrorMessage::ID, StreamBuffer(ErrorMessage(ErrorMessage::ID, 0, VAL_INIT)), false);
		}
	}

	Client::Client(const IpPort &addr, boost::uint64_t keepAliveTimeout, bool useSsl)
		: TcpClientBase(addr, useSsl)
		, m_keepAliveTimeout(keepAliveTimeout)
		, m_payloadLen((boost::uint64_t)-1), m_messageId(0)
	{
	}
	Client::~Client(){
		if(m_payloadLen != (boost::uint64_t)-1){
			LOG_POSEIDON_WARNING("Now that this session is to be destroyed, a premature response has to be discarded.");
		}
	}

	void Client::onReadAvail(const void *data, std::size_t size){
		PROFILE_ME;

		try {
			if((m_keepAliveTimeout != 0) && !m_keepAliveTimer){
				m_keepAliveTimer = TimerDaemon::registerTimer(m_keepAliveTimeout, m_keepAliveTimeout,
					boost::bind(&keepAliveTimerProc, virtualWeakFromThis<Client>()));
			}

			m_payload.put(data, size);
			for(;;){
				if(m_payloadLen == (boost::uint64_t)-1){
					boost::uint16_t messageId;
					boost::uint64_t payloadLen;
					if(!MessageBase::decodeHeader(messageId, payloadLen, m_payload)){
						break;
					}
					m_messageId = messageId;
					m_payloadLen = payloadLen;
					LOG_POSEIDON_DEBUG("Message id = ", m_messageId, ", len = ", m_payloadLen);
				}
				if(m_payload.size() < m_payloadLen){
					break;
				}
				enqueueJob(boost::make_shared<ServerResponseJob>(
					virtualWeakFromThis<Client>(), m_messageId, m_payload.cut(m_payloadLen)));
				m_payloadLen = (boost::uint64_t)-1;
				m_messageId = 0;
			}
		} catch(Exception &e){
			LOG_POSEIDON_ERROR("Cbpp::Exception thrown while parsing data, message id = ", m_messageId,
				", statusCode = ", static_cast<int>(e.statusCode()), ", what = ", e.what());
			throw;
		} catch(...){
			LOG_POSEIDON_ERROR("Forwarding exception... message id = ", m_messageId);
			throw;
		}
	}

	bool Client::send(boost::uint16_t messageId, StreamBuffer contents, bool fin){
		StreamBuffer data;
		MessageBase::encodeHeader(data, messageId, contents.size());
		data.splice(contents);
		return TcpSessionBase::send(STD_MOVE(data), fin);
	}
}

}
