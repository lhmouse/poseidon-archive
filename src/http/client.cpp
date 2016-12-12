// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2016, LH_Mouse. All wrongs reserved.

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
		const boost::weak_ptr<TcpSessionBase> m_category;
		const boost::weak_ptr<Client> m_weak_client;

	protected:
		explicit SyncJobBase(const boost::shared_ptr<Client> &client)
			: m_guard(client), m_category(client), m_weak_client(client)
		{
		}

	private:
		boost::weak_ptr<const void> get_category() const FINAL {
			return m_category;
		}
		void perform() FINAL {
			PROFILE_ME;

			const AUTO(client, m_weak_client.lock());
			if(!client || client->has_been_shutdown_read()){
				return;
			}

			try {
				really_perform(client);
			} catch(Exception &e){
				LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO,
					"Http::Exception thrown in HTTP servlet: status_code = ", e.get_status_code(), ", what = ", e.what());
				client->shutdown_read();
				client->shutdown_write();
				throw;
			} catch(std::exception &e){
				LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "std::exception thrown: what = ", e.what());
				client->shutdown_read();
				client->shutdown_write();
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

	class Client::ReadHupJob : public Client::SyncJobBase {
	public:
		explicit ReadHupJob(const boost::shared_ptr<Client> &client)
			: SyncJobBase(client)
		{
		}

	protected:
		void really_perform(const boost::shared_ptr<Client> &client) OVERRIDE {
			PROFILE_ME;

			client->shutdown_write();
		}
	};

	class Client::ResponseJob : public Client::SyncJobBase {
	private:
		ResponseHeaders m_response_headers;
		StreamBuffer m_entity;

	public:
		ResponseJob(const boost::shared_ptr<Client> &client,
			ResponseHeaders response_headers, StreamBuffer entity)
			: SyncJobBase(client)
			, m_response_headers(STD_MOVE(response_headers)), m_entity(STD_MOVE(entity))
		{
		}

	protected:
		void really_perform(const boost::shared_ptr<Client> &client) OVERRIDE {
			PROFILE_ME;

			client->on_sync_response(STD_MOVE(m_response_headers), STD_MOVE(m_entity));
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
			boost::make_shared<ConnectJob>(virtual_shared_from_this<Client>()),
			VAL_INIT);

		LowLevelClient::on_connect();
	}
	void Client::on_read_hup() NOEXCEPT
	try {
		PROFILE_ME;

		JobDispatcher::enqueue(
			boost::make_shared<ReadHupJob>(virtual_shared_from_this<Client>()),
			VAL_INIT);

		LowLevelClient::on_read_hup();
	} catch(std::exception &e){
		LOG_POSEIDON_WARNING("std::exception thrown: what = ", e.what());
		force_shutdown();
	} catch(...){
		LOG_POSEIDON_WARNING("Unknown exception thrown.");
		force_shutdown();
	}

	void Client::on_low_level_response_headers(ResponseHeaders response_headers, boost::uint64_t content_length){
		PROFILE_ME;

		(void)content_length;

		m_response_headers = STD_MOVE(response_headers);
		m_entity.clear();
	}
	void Client::on_low_level_response_entity(boost::uint64_t entity_offset, StreamBuffer entity){
		PROFILE_ME;

		(void)entity_offset;

		m_entity.splice(entity);
	}
	boost::shared_ptr<UpgradedClientBase> Client::on_low_level_response_end(boost::uint64_t content_length, OptionalMap headers){
		PROFILE_ME;

		(void)content_length;

		for(AUTO(it, headers.begin()); it != headers.end(); ++it){
			m_response_headers.headers.append(it->first, STD_MOVE(it->second));
		}

		JobDispatcher::enqueue(
			boost::make_shared<ResponseJob>(virtual_shared_from_this<Client>(),
				STD_MOVE(m_response_headers), STD_MOVE(m_entity)),
			VAL_INIT);

		return VAL_INIT;
	}

	void Client::on_sync_connect(){
		PROFILE_ME;
		LOG_POSEIDON_INFO("CBPP client connected: remote = ", get_remote_info());
	}
}

}
