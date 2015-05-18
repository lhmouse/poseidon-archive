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
				} catch(Exception &e){
					LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO,
						"Cbpp::Exception thrown: statusCode = ", e.statusCode(), ", what = ", e.what());
					client->forceShutdown();
					throw;
				} catch(std::exception &e){
					LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO,
						"std::exception thrown: what = ", e.what());
					client->forceShutdown();
					throw;
				} catch(...){
					LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO,
						"Unknown exception thrown.");
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

			client->sendControl(CTL_HEARTBEAT, 0, VAL_INIT);
		}
	};

	class Client::DataMessageHeaderJob : public ClientJobBase {
	private:
		const unsigned m_messageId;
		const boost::uint64_t m_payloadSize;

	public:
		DataMessageHeaderJob(const boost::shared_ptr<Client> &client, unsigned messageId, boost::uint64_t payloadSize)
			: ClientJobBase(client)
			, m_messageId(messageId), m_payloadSize(payloadSize)
		{
		}

	protected:
		void perform(const boost::shared_ptr<Client> &client) const OVERRIDE {
			PROFILE_ME;

			client->onSyncDataMessageHeader(m_messageId, m_payloadSize);
		}
	};

	class Client::DataMessagePayloadJob : public ClientJobBase {
	public:
		const boost::uint64_t m_payloadOffset;
		const StreamBuffer m_payload;

	public:
		DataMessagePayloadJob(const boost::shared_ptr<Client> &client, boost::uint64_t payloadOffset, StreamBuffer payload)
			: ClientJobBase(client)
			, m_payloadOffset(payloadOffset), m_payload(STD_MOVE(payload))
		{
		}

	protected:
		void perform(const boost::shared_ptr<Client> &client) const OVERRIDE {
			PROFILE_ME;

			client->onSyncDataMessagePayload(m_payloadOffset, m_payload);
		}
	};

	class Client::DataMessageEndJob : public ClientJobBase {
	public:
		const boost::uint64_t m_payloadSize;

	public:
		DataMessageEndJob(const boost::shared_ptr<Client> &client, boost::uint64_t payloadSize)
			: ClientJobBase(client)
			, m_payloadSize(payloadSize)
		{
		}

	protected:
		void perform(const boost::shared_ptr<Client> &client) const OVERRIDE {
			PROFILE_ME;

			client->onSyncDataMessageEnd(m_payloadSize);
		}
	};

	class Client::ErrorMessageJob : public ClientJobBase {
	private:
		const boost::uint16_t m_messageId;
		const StatusCode m_statusCode;
		const std::string m_reason;

	public:
		ErrorMessageJob(const boost::shared_ptr<Client> &client, boost::uint16_t messageId, StatusCode statusCode, std::string reason)
			: ClientJobBase(client)
			, m_messageId(messageId), m_statusCode(statusCode), m_reason(STD_MOVE(reason))
		{
		}

	protected:
		void perform(const boost::shared_ptr<Client> &client) const OVERRIDE {
			PROFILE_ME;

			client->onSyncErrorMessage(m_messageId, m_statusCode, m_reason);
		}
	};

	Client::Client(const SockAddr &addr, bool useSsl, boost::uint64_t keepAliveInterval)
		: TcpClientBase(addr, useSsl)
		, m_keepAliveInterval(keepAliveInterval)
	{
	}
	Client::Client(const IpPort &addr, bool useSsl, boost::uint64_t keepAliveInterval)
		: TcpClientBase(addr, useSsl)
		, m_keepAliveInterval(keepAliveInterval)
	{
	}
	Client::~Client(){
	}

	void Client::onReadAvail(const void *data, std::size_t size){
		PROFILE_ME;

		Reader::putEncodedData(StreamBuffer(data, size));
	}

	void Client::onDataMessageHeader(boost::uint16_t messageId, boost::uint64_t payloadSize){
		PROFILE_ME;

		enqueueJob(boost::make_shared<DataMessageHeaderJob>(
			virtualSharedFromThis<Client>(), messageId, payloadSize));
	}
	void Client::onDataMessagePayload(boost::uint64_t payloadOffset, StreamBuffer payload){
		PROFILE_ME;

		enqueueJob(boost::make_shared<DataMessagePayloadJob>(
			virtualSharedFromThis<Client>(), payloadOffset, STD_MOVE(payload)));
	}
	bool Client::onDataMessageEnd(boost::uint64_t payloadSize){
		PROFILE_ME;

		enqueueJob(boost::make_shared<DataMessageEndJob>(
			virtualSharedFromThis<Client>(), payloadSize));

		return true;
	}

	bool Client::onControlMessage(ControlCode controlCode, boost::int64_t vintParam, std::string stringParam){
		PROFILE_ME;

		enqueueJob(boost::make_shared<ErrorMessageJob>(
			virtualSharedFromThis<Client>(), controlCode, vintParam, STD_MOVE(stringParam)));

		return true;
	}

	long Client::onEncodedDataAvail(StreamBuffer encoded){
		PROFILE_ME;

		return TcpSessionBase::send(STD_MOVE(encoded));
	}

	void Client::onSyncErrorMessage(boost::uint16_t messageId, StatusCode statusCode, const std::string &reason){
		PROFILE_ME;

		LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO,
			"Received CBPP error message from server: messageId = ", messageId, ", statusCode = ", statusCode, ", reason = ", reason);
	}

	bool Client::send(boost::uint16_t messageId, StreamBuffer payload){
		PROFILE_ME;

		return Writer::putDataMessage(messageId, STD_MOVE(payload));
	}
	bool Client::sendControl(ControlCode controlCode, boost::int64_t vintParam, std::string stringParam){
		PROFILE_ME;

		return Writer::putControlMessage(controlCode, vintParam, STD_MOVE(stringParam));
	}
}

}
