// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "client.hpp"
#include "exception.hpp"
#include "control_message.hpp"
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
	}

	class Client::KeepAliveJob : public ClientJobBase {
	public:
		explicit KeepAliveJob(const boost::shared_ptr<Client> &client)
			: ClientJobBase(client)
		{
		}

	protected:
		void perform(const boost::shared_ptr<Client> &client) const OVERRIDE {
			PROFILE_ME;

			client->send(ControlMessage(ControlMessage::ID, 0, VAL_INIT));
		}
	};

	class Client::ResponseJob : public ClientJobBase {
	private:
		const unsigned m_messageId;
		const boost::uint64_t m_payloadLen;

	public:
		ResponseJob(const boost::shared_ptr<Client> &client, unsigned messageId, boost::uint64_t payloadLen)
			: ClientJobBase(client)
			, m_messageId(messageId), m_payloadLen(payloadLen)
		{
		}

	protected:
		void perform(const boost::shared_ptr<Client> &client) const OVERRIDE {
			PROFILE_ME;

			LOG_POSEIDON_DEBUG("Response: messageId = ", m_messageId, ", payloadLen = ", m_payloadLen);
			client->onResponse(m_messageId, m_payloadLen);
		}
	};

	class Client::PayloadJob : public ClientJobBase {
	public:
		const boost::uint64_t m_payloadOffset;
		const StreamBuffer m_payload;

	public:
		PayloadJob(const boost::shared_ptr<Client> &client, boost::uint64_t payloadOffset, StreamBuffer payload)
			: ClientJobBase(client)
			, m_payloadOffset(payloadOffset), m_payload(STD_MOVE(payload))
		{
		}

	protected:
		void perform(const boost::shared_ptr<Client> &client) const OVERRIDE {
			PROFILE_ME;

			LOG_POSEIDON_DEBUG("Payload: payloadOffset = ", m_payloadOffset, ", segmentSize = ", m_payload.size());
			client->onPayload(m_payloadOffset, m_payload);
		}
	};

	class Client::ControlJob : public ClientJobBase {
	private:
		const ControlMessage m_msg;

	public:
		ControlJob(const boost::shared_ptr<Client> &client, ControlMessage msg)
			: ClientJobBase(client)
			, m_msg(STD_MOVE(msg))
		{
		}

	protected:
		void perform(const boost::shared_ptr<Client> &client) const OVERRIDE {
			PROFILE_ME;

			LOG_POSEIDON_DEBUG("Control: msg = ", m_msg);
			client->onControl(m_msg.messageId, static_cast<StatusCode>(m_msg.statusCode), m_msg.reason);
		}
	};

	Client::Client(const IpPort &addr, boost::uint64_t keepAliveTimeout, bool useSsl)
		: TcpClientBase(addr, useSsl)
		, m_keepAliveTimeout(keepAliveTimeout)
		, m_sizeExpecting(2), m_state(S_PAYLOAD_LEN)
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
				if(m_received.size() < m_sizeExpecting){
					break;
				}

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
					m_payloadOffset = 0;

					enqueueJob(boost::make_shared<ResponseJob>(
						virtualSharedFromThis<Client>(), m_messageId, m_payloadLen));

					if(m_messageId != ControlMessage::ID){
						m_sizeExpecting = std::min<boost::uint64_t>(m_payloadLen, 1024);
					} else {
						m_sizeExpecting = m_payloadLen;
					}
					m_state = S_PAYLOAD;
					break;

				case S_PAYLOAD:
					if(m_messageId != ControlMessage::ID){
						const AUTO(bytesAvail, std::min<boost::uint64_t>(m_received.size(), m_payloadLen - m_payloadOffset));
						enqueueJob(boost::make_shared<PayloadJob>(
							virtualSharedFromThis<Client>(), m_payloadOffset, m_received.cut(bytesAvail)));
						m_payloadOffset += bytesAvail;

						if(m_payloadLen > m_payloadOffset){
							m_sizeExpecting = std::min<boost::uint64_t>(m_payloadLen - m_payloadOffset, 1024);
							// m_state = S_PAYLOAD;
							break;
						}
					} else {
						enqueueJob(boost::make_shared<ControlJob>(
							virtualSharedFromThis<Client>(), ControlMessage(m_received.cut(m_payloadLen))));
					}

					m_messageId = 0;
					m_payloadLen = 0;

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

	bool Client::send(boost::uint16_t messageId, StreamBuffer payload, bool fin){
		PROFILE_ME;

		LOG_POSEIDON_DEBUG("Sending frame: messageId = ", messageId, ", size = ", payload.size(), ", fin = ", fin);
		StreamBuffer frame;
		boost::uint16_t temp16;
		boost::uint64_t temp64;
		if(payload.size() < 0xFFFF){
			storeLe(temp16, payload.size());
			frame.put(&temp16, 2);
		} else {
			storeLe(temp16, 0xFFFF);
			frame.put(&temp16, 2);
			storeLe(temp64, payload.size());
			frame.put(&temp64, 8);
		}
		storeLe(temp16, messageId);
		frame.put(&temp16, 2);
		frame.splice(payload);
		return TcpSessionBase::send(STD_MOVE(frame), fin);
	}
}

}
