#include "../precompiled.hpp"
#include "client.hpp"
#include "exception.hpp"
#include "status_codes.hpp"
#include "../singletons/job_dispatcher.hpp"
#include "../log.hpp"
#include "../job_base.hpp"
#include "../profiler.hpp"

namespace Poseidon {

namespace Http {
	class Client::SyncJobBase : public JobBase {
	private:
		const TcpSessionBase::DelayedShutdownGuard m_guard;
		const boost::weak_ptr<Client> m_client;

	protected:
		explicit SyncJobBase(const boost::shared_ptr<Client> &client)
			: m_guard(client), m_client(client)
		{
		}

	private:
		boost::weak_ptr<const void> get_category() const FINAL {
			return m_client;
		}
		void perform() FINAL {
			PROFILE_ME;

			const AUTO(client, m_client.lock());
			if(!client){
				return;
			}

			try {
				really_perform(client);
			} catch(std::exception &e){
				LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "std::exception thrown: what = ", e.what());
				client->force_shutdown();
				throw;
			} catch(...){
				LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "Unknown exception thrown.");
				client->force_shutdown();
				throw;
			}
		}

	protected:
		virtual void really_perform(const boost::shared_ptr<Client> &client) = 0;
	};

	class Client::ConnectJob : public Client::SyncJobBase {
	public:
		explicit ConnectJob(const boost::shared_ptr<Client> &client)
			: SyncJobBase(client)
		{
		}

	protected:
		void really_perform(const boost::shared_ptr<Client> &client) OVERRIDE {
			PROFILE_ME;

			client->on_sync_connect();
		}
	};

	class Client::ResponseHeadersJob : public Client::SyncJobBase {
	private:
		ResponseHeaders m_response_headers;
		std::string m_transfer_encoding;
		boost::uint64_t m_content_length;

	public:
		ResponseHeadersJob(const boost::shared_ptr<Client> &client,
			ResponseHeaders response_headers, std::string transfer_encoding, boost::uint64_t content_length)
			: SyncJobBase(client)
			, m_response_headers(STD_MOVE(response_headers)), m_transfer_encoding(STD_MOVE(transfer_encoding)), m_content_length(content_length)
		{
		}

	protected:
		void really_perform(const boost::shared_ptr<Client> &client) OVERRIDE {
			PROFILE_ME;

			client->on_sync_response_headers(STD_MOVE(m_response_headers), STD_MOVE(m_transfer_encoding), m_content_length);
		}
	};

	class Client::ResponseEntityJob : public Client::SyncJobBase {
	private:
		boost::uint64_t m_content_offset;
		bool m_is_chunked;
		StreamBuffer m_entity;

	public:
		ResponseEntityJob(const boost::shared_ptr<Client> &client, boost::uint64_t content_offset, bool is_chunked, StreamBuffer entity)
			: SyncJobBase(client)
			, m_content_offset(content_offset), m_is_chunked(is_chunked), m_entity(STD_MOVE(entity))
		{
		}

	protected:
		void really_perform(const boost::shared_ptr<Client> &client) OVERRIDE {
			PROFILE_ME;

			client->on_sync_response_entity(m_content_offset, m_is_chunked, STD_MOVE(m_entity));
		}
	};

	class Client::ResponseEndJob : public Client::SyncJobBase {
	private:
		boost::uint64_t m_content_length;
		bool m_is_chunked;
		OptionalMap m_headers;

	public:
		ResponseEndJob(const boost::shared_ptr<Client> &client, boost::uint64_t content_length, bool is_chunked, OptionalMap headers)
			: SyncJobBase(client)
			, m_content_length(content_length), m_is_chunked(is_chunked), m_headers(STD_MOVE(headers))
		{
		}

	protected:
		void really_perform(const boost::shared_ptr<Client> &client) OVERRIDE {
			PROFILE_ME;

			client->on_sync_response_end(m_content_length, m_is_chunked, STD_MOVE(m_headers));
		}
	};

	Client::Client(const SockAddr &addr, bool use_ssl)
		: LowLevelClient(addr, use_ssl)
	{
	}
	Client::Client(const IpPort &addr, bool use_ssl)
		: LowLevelClient(addr, use_ssl)
	{
	}
	Client::~Client(){
	}

	void Client::on_connect(){
		PROFILE_ME;

		JobDispatcher::enqueue(
			boost::make_shared<ConnectJob>(
				virtual_shared_from_this<Client>()),
			VAL_INIT);

		LowLevelClient::on_connect();
	}

	void Client::on_low_level_response_headers(ResponseHeaders response_headers,
		std::string transfer_encoding, boost::uint64_t content_length)
	{
		PROFILE_ME;

		JobDispatcher::enqueue(
			boost::make_shared<ResponseHeadersJob>(
				virtual_shared_from_this<Client>(), STD_MOVE(response_headers), STD_MOVE(transfer_encoding), content_length),
			VAL_INIT);
	}
	void Client::on_low_level_response_entity(boost::uint64_t entity_offset, bool is_chunked, StreamBuffer entity){
		PROFILE_ME;

		JobDispatcher::enqueue(
			boost::make_shared<ResponseEntityJob>(
				virtual_shared_from_this<Client>(), entity_offset, is_chunked, STD_MOVE(entity)),
			VAL_INIT);
	}
	bool Client::on_low_level_response_end(boost::uint64_t content_length, bool is_chunked, OptionalMap headers){
		PROFILE_ME;

		JobDispatcher::enqueue(
			boost::make_shared<ResponseEndJob>(
				virtual_shared_from_this<Client>(), content_length, is_chunked, STD_MOVE(headers)),
			VAL_INIT);

		return true;
	}

	void Client::on_sync_connect(){
		PROFILE_ME;
		LOG_POSEIDON_INFO("CBPP client connected: remote = ", get_remote_info());
	}
}

}
