// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "client.hpp"
#include "exception.hpp"
#include "control_message.hpp"
#include "../singletons/timer_daemon.hpp"
#include "../log.hpp"
#include "../job_base.hpp"
#include "../profiler.hpp"

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
		const boost::uint16_t m_messageId;
		const StatusCode m_statusCode;
		const std::string m_reason;

	public:
		ControlJob(const boost::shared_ptr<Client> &client,
			boost::uint16_t messageId, StatusCode statusCode, std::string reason)
			: ClientJobBase(client)
			, m_messageId(messageId), m_statusCode(statusCode), m_reason(STD_MOVE(reason))
		{
		}

	protected:
		void perform(const boost::shared_ptr<Client> &client) const OVERRIDE {
			PROFILE_ME;

			LOG_POSEIDON_DEBUG("Control message: messageId = ", m_messageId,
				", statusCode = ", m_statusCode, ", reason = ", m_reason);
			client->onControl(m_messageId, m_statusCode, m_reason);
		}
	};

	Client::Client(const IpPort &addr, boost::uint64_t keepAliveTimeout, bool useSsl)
		: LowLevelClient(addr, keepAliveTimeout, useSsl)
	{
	}
	Client::~Client(){
	}

	void Client::onLowLevelResponse(boost::uint16_t messageId, boost::uint64_t payloadLen){
		PROFILE_ME;

		enqueueJob(boost::make_shared<ResponseJob>(
			virtualSharedFromThis<Client>(), messageId, payloadLen));
	}
	void Client::onLowLevelPayload(boost::uint64_t payloadOffset, StreamBuffer payload){
		PROFILE_ME;

		enqueueJob(boost::make_shared<PayloadJob>(
			virtualSharedFromThis<Client>(), payloadOffset, STD_MOVE(payload)));
	}

	void Client::onLowLevelControl(boost::uint16_t messageId, StatusCode statusCode, std::string reason){
		PROFILE_ME;

		enqueueJob(boost::make_shared<ControlJob>(
			virtualSharedFromThis<Client>(), messageId, statusCode, STD_MOVE(reason)));
	}
}

}
