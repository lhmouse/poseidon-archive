// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2016, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "client.hpp"
#include "exception.hpp"
#include "control_message.hpp"
#include "../singletons/timer_daemon.hpp"
#include "../log.hpp"
#include "../job_base.hpp"
#include "../profiler.hpp"
#include "../time.hpp"

namespace Poseidon {

namespace Cbpp {
	class Client::SyncJobBase : public JobBase {
	private:
		const boost::weak_ptr<Client> m_client;

	protected:
		explicit SyncJobBase(const boost::shared_ptr<Client> &client)
			: m_client(client)
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
			} catch(Exception &e){
				LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO,
					"Cbpp::Exception thrown: status_code = ", e.status_code(), ", what = ", e.what());
				client->force_shutdown();
				throw;
			} catch(std::exception &e){
				LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO,
					"std::exception thrown: what = ", e.what());
				client->force_shutdown();
				throw;
			} catch(...){
				LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO,
					"Unknown exception thrown.");
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

	class Client::DataMessageHeaderJob : public Client::SyncJobBase {
	private:
		unsigned m_message_id;
		boost::uint64_t m_payload_size;

	public:
		DataMessageHeaderJob(const boost::shared_ptr<Client> &client, unsigned message_id, boost::uint64_t payload_size)
			: SyncJobBase(client)
			, m_message_id(message_id), m_payload_size(payload_size)
		{
		}

	protected:
		void really_perform(const boost::shared_ptr<Client> &client) OVERRIDE {
			PROFILE_ME;

			client->on_sync_data_message_header(m_message_id, m_payload_size);
		}
	};

	class Client::DataMessagePayloadJob : public Client::SyncJobBase {
	public:
		boost::uint64_t m_payload_offset;
		StreamBuffer m_payload;

	public:
		DataMessagePayloadJob(const boost::shared_ptr<Client> &client, boost::uint64_t payload_offset, StreamBuffer payload)
			: SyncJobBase(client)
			, m_payload_offset(payload_offset), m_payload(STD_MOVE(payload))
		{
		}

	protected:
		void really_perform(const boost::shared_ptr<Client> &client) OVERRIDE {
			PROFILE_ME;

			client->on_sync_data_message_payload(m_payload_offset, STD_MOVE(m_payload));
		}
	};

	class Client::DataMessageEndJob : public Client::SyncJobBase {
	public:
		boost::uint64_t m_payload_size;

	public:
		DataMessageEndJob(const boost::shared_ptr<Client> &client, boost::uint64_t payload_size)
			: SyncJobBase(client)
			, m_payload_size(payload_size)
		{
		}

	protected:
		void really_perform(const boost::shared_ptr<Client> &client) OVERRIDE {
			PROFILE_ME;

			client->on_sync_data_message_end(m_payload_size);

			client->m_last_pong_time = get_fast_mono_clock();
		}
	};

	class Client::ErrorMessageJob : public Client::SyncJobBase {
	private:
		boost::uint16_t m_message_id;
		StatusCode m_status_code;
		std::string m_reason;

	public:
		ErrorMessageJob(const boost::shared_ptr<Client> &client, boost::uint16_t message_id, StatusCode status_code, std::string reason)
			: SyncJobBase(client)
			, m_message_id(message_id), m_status_code(status_code), m_reason(STD_MOVE(reason))
		{
		}

	protected:
		void really_perform(const boost::shared_ptr<Client> &client) OVERRIDE {
			PROFILE_ME;

			client->on_sync_error_message(m_message_id, m_status_code, STD_MOVE(m_reason));

			client->m_last_pong_time = get_fast_mono_clock();
		}
	};

	void Client::keep_alive_timer_proc(const boost::weak_ptr<Client> &weak_client, boost::uint64_t now, boost::uint64_t period){
		PROFILE_ME;

		const AUTO(client, weak_client.lock());
		if(!client){
			return;
		}

		if(client->m_last_pong_time < now - period * 2){
			LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO,
				"No pong received since the last two keep alive intervals. Shut down the connection.");
			client->force_shutdown();
			return;
		}

		client->send_control(CTL_PING, 0, boost::lexical_cast<std::string>(get_utc_time()));
	}

	Client::Client(const SockAddr &addr, bool use_ssl, boost::uint64_t keep_alive_interval)
		: TcpClientBase(addr, use_ssl)
		, m_keep_alive_interval(keep_alive_interval)
		, m_last_pong_time((boost::uint64_t)-1)
	{
	}
	Client::Client(const IpPort &addr, bool use_ssl, boost::uint64_t keep_alive_interval)
		: TcpClientBase(addr, use_ssl)
		, m_keep_alive_interval(keep_alive_interval)
		, m_last_pong_time((boost::uint64_t)-1)
	{
	}
	Client::~Client(){
	}

	void Client::on_connect(){
		PROFILE_ME;

		enqueue_job(boost::make_shared<ConnectJob>(
			virtual_shared_from_this<Client>()));

		TcpClientBase::on_connect();
	}
	void Client::on_read_avail(StreamBuffer data){
		PROFILE_ME;

		Reader::put_encoded_data(STD_MOVE(data));
	}

	void Client::on_data_message_header(boost::uint16_t message_id, boost::uint64_t payload_size){
		PROFILE_ME;

		enqueue_job(boost::make_shared<DataMessageHeaderJob>(
			virtual_shared_from_this<Client>(), message_id, payload_size));
	}
	void Client::on_data_message_payload(boost::uint64_t payload_offset, StreamBuffer payload){
		PROFILE_ME;

		enqueue_job(boost::make_shared<DataMessagePayloadJob>(
			virtual_shared_from_this<Client>(), payload_offset, STD_MOVE(payload)));
	}
	bool Client::on_data_message_end(boost::uint64_t payload_size){
		PROFILE_ME;

		enqueue_job(boost::make_shared<DataMessageEndJob>(
			virtual_shared_from_this<Client>(), payload_size));

		return true;
	}

	bool Client::on_control_message(ControlCode control_code, boost::int64_t vint_param, std::string string_param){
		PROFILE_ME;

		enqueue_job(boost::make_shared<ErrorMessageJob>(
			virtual_shared_from_this<Client>(), control_code, vint_param, STD_MOVE(string_param)));

		return true;
	}

	long Client::on_encoded_data_avail(StreamBuffer encoded){
		PROFILE_ME;

		if(!m_keep_alive_timer){
			m_keep_alive_timer = TimerDaemon::register_timer(m_keep_alive_interval, m_keep_alive_interval,
				boost::bind(&keep_alive_timer_proc, virtual_weak_from_this<Client>(), _2, _3));
		}

		return TcpClientBase::send(STD_MOVE(encoded));
	}

	void Client::on_sync_connect(){
		PROFILE_ME;
		LOG_POSEIDON_INFO("CBPP client connected: remote = ", get_remote_info());
	}

	void Client::on_sync_error_message(boost::uint16_t message_id, StatusCode status_code, std::string reason){
		PROFILE_ME;
		LOG_POSEIDON_TRACE("Received CBPP error message from server: message_id = ", message_id,
			", status_code = ", status_code, ", reason = ", reason);

		if(status_code < 0){
			LOG_POSEIDON_WARNING("Fatal CBPP error: status_code = ", status_code);

			force_shutdown();
		}
	}

	bool Client::send(boost::uint16_t message_id, StreamBuffer payload){
		PROFILE_ME;

		return Writer::put_data_message(message_id, STD_MOVE(payload));
	}
	bool Client::send_control(ControlCode control_code, boost::int64_t vint_param, std::string string_param){
		PROFILE_ME;

		return Writer::put_control_message(control_code, vint_param, STD_MOVE(string_param));
	}
}

}
