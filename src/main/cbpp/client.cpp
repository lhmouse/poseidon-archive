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
#include "../endian.hpp"

namespace Poseidon {

namespace Cbpp {
	namespace {
		class ClientJobBase : public JobBase {
		private:
			const boost::weak_ptr<Client> m_client;

		protected:
			explicit ClientJobBase(const boost::shared_ptr<Client> &client)
				: m_client(client)
			{
			}

		protected:
			virtual void perform(const boost::shared_ptr<Client> &client) const = 0;

		private:
			boost::weak_ptr<const void> getCategory() const FINAL {
				return m_client;
			}
			void perform() const FINAL {
				PROFILE_ME;

				const AUTO(client, m_client.lock());
				if(!client){
					return;
				}

				try {
					perform(client);
				} catch(TryAgainLater &){
					throw;
				} catch(std::exception &e){
					LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "std::exception thrown: what = ", e.what());
					client->forceShutdown();
					throw;
				} catch(...){
					LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "Unknown exception thrown.");
					client->forceShutdown();
					throw;
				}
			}
		};

		class ResponseJob : public ClientJobBase {
		private:
			const unsigned m_messageId;
			const StreamBuffer m_payload;

		public:
			ResponseJob(const boost::shared_ptr<Client> &client, unsigned messageId, StreamBuffer payload)
				: ClientJobBase(client)
				, m_messageId(messageId), m_payload(STD_MOVE(payload))
			{
			}

		protected:
			void perform(const boost::shared_ptr<Client> &client) const OVERRIDE {
				PROFILE_ME;

				if(m_messageId == ErrorMessage::ID){
					AUTO(payload, m_payload);
					ErrorMessage req(payload);
					LOG_POSEIDON_DEBUG("Dispatching error message: ", req);
					client->onError(req.messageId, req.statusCode, STD_MOVE(req.reason));
				} else {
					LOG_POSEIDON_DEBUG("Dispatching message: messageId = ", m_messageId, ", payload size = ", m_payload.size());
					client->onResponse(m_messageId, m_payload);
				}
			}
		};

		class KeepAliveJob : public ClientJobBase {
		public:
			explicit KeepAliveJob(const boost::shared_ptr<Client> &client)
				: ClientJobBase(client)
			{
			}

		protected:
			void perform(const boost::shared_ptr<Client> &client) const OVERRIDE {
				PROFILE_ME;

				client->send(ErrorMessage(ErrorMessage::ID, 0, VAL_INIT));
			}
		};
	}

	Client::Client(const IpPort &addr, boost::uint64_t keepAliveTimeout, bool useSsl)
		: TcpClientBase(addr, useSsl)
		, m_keepAliveTimeout(keepAliveTimeout)
	{
	}
	Client::~Client(){
		if(m_state != S_PAYLOAD_LEN){
			LOG_POSEIDON_WARNING("Now that this client is to be destroyed, a premature response has to be discarded.");
		}
	}

	void Client::onReadAvail(const void *data, std::size_t size){
		PROFILE_ME;

		try {
			m_received.put(data, size);

			for(;;){
				boost::uint64_t sizeTotal;
				bool gotExpected;
				if(m_received.size() < m_sizeExpecting){
					sizeTotal = m_sizeTotal + m_received.size();
					gotExpected = false;
				} else {
					sizeTotal = m_sizeTotal + m_sizeExpecting;
					gotExpected = true;
				}
				if(!gotExpected){
					break;
				}
				m_sizeTotal = sizeTotal;

				switch(m_state){
					boost::uint16_t temp16;
					boost::uint64_t temp64;

				case S_PAYLOAD_LEN:
					m_received.get(&temp16, 2);
					m_payloadLen = loadLe(temp16);
					if(m_payloadLen == 0xFFFF){
						m_sizeExpecting = 8;
						m_state = S_EX_PAYLOAD_LEN;
					} else {
						m_sizeExpecting = 2;
						m_state = S_MESSAGE_ID;
					}
					break;

				case S_EX_PAYLOAD_LEN:
					m_received.get(&temp64, 8);
					m_payloadLen = loadLe(temp64);

					m_sizeExpecting = 2;
					m_state = S_MESSAGE_ID;
					break;

				case S_MESSAGE_ID:
					LOG_POSEIDON_DEBUG("Payload length = ", m_payloadLen);

					m_received.get(&temp16, 2);
					m_messageId = loadLe(temp16);

					m_sizeExpecting = m_payloadLen;
					m_state = S_PAYLOAD;
					break;

				case S_PAYLOAD:
					enqueueJob(boost::make_shared<ResponseJob>(
						virtualSharedFromThis<Client>(), m_messageId, m_received.cut(m_payloadLen)));

					m_messageId = 0;
					m_payloadLen = 0;

					m_sizeTotal = 0;
					m_sizeExpecting = 2;
					m_state = S_PAYLOAD_LEN;
					break;

				default:
					LOG_POSEIDON_FATAL("Invalid state: ", static_cast<unsigned>(m_state));
					std::abort();
				}
			}
		} catch(std::exception &e){
			LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "std::exception thrown while parsing data, message id = ", m_messageId,
				", what = ", e.what());
			forceShutdown();
		}
	}

	void Client::onError(ControlCode controlCode, StatusCode statusCode, std::string reason){
		(void)controlCode;
		(void)statusCode;
		(void)reason;
	}

	bool Client::send(boost::uint16_t messageId, StreamBuffer contents, bool fin){
		PROFILE_ME;

		LOG_POSEIDON_DEBUG("Sending frame: messageId = ", messageId, ", size = ", contents.size(), ", fin = ", fin);
		StreamBuffer frame;
		boost::uint16_t temp16;
		boost::uint64_t temp64;
		if(contents.size() < 0xFFFF){
			storeLe(temp16, contents.size());
			frame.put(&temp16, 2);
		} else {
			storeLe(temp16, 0xFFFF);
			frame.put(&temp16, 2);
			storeLe(temp64, contents.size());
			frame.put(&temp64, 8);
		}
		storeLe(temp16, messageId);
		frame.put(&temp16, 2);
		frame.splice(contents);
		return TcpSessionBase::send(STD_MOVE(frame), fin);
	}
}

}
