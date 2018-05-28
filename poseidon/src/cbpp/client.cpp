// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "client.hpp"
#include "exception.hpp"
#include "../singletons/job_dispatcher.hpp"
#include "../log.hpp"
#include "../job_base.hpp"
#include "../profiler.hpp"
#include "../time.hpp"

namespace Poseidon {
namespace Cbpp {

class Client::Sync_job_base : public Job_base {
private:
	const Socket_base::Delayed_shutdown_guard m_guard;
	const boost::weak_ptr<Client> m_weak_client;

protected:
	explicit Sync_job_base(const boost::shared_ptr<Client> &client)
		: m_guard(client), m_weak_client(client)
	{
		//
	}

private:
	boost::weak_ptr<const void> get_category() const FINAL {
		return m_weak_client;
	}
	void perform() FINAL {
		POSEIDON_PROFILE_ME;

		const AUTO(client, m_weak_client.lock());
		if(!client || client->has_been_shutdown_write()){
			return;
		}

		try {
			really_perform(client);
		} catch(Exception &e){
			POSEIDON_LOG(Logger::special_major | Logger::level_info, "Cbpp::Exception thrown: status_code = ", e.get_status_code(), ", what = ", e.what());
			client->force_shutdown();
		} catch(std::exception &e){
			POSEIDON_LOG(Logger::special_major | Logger::level_info, "std::exception thrown: what = ", e.what());
			client->force_shutdown();
		} catch(...){
			POSEIDON_LOG(Logger::special_major | Logger::level_info, "Unknown exception thrown.");
			client->force_shutdown();
		}
	}

protected:
	virtual void really_perform(const boost::shared_ptr<Client> &client) = 0;
};

class Client::Connect_job : public Client::Sync_job_base {
public:
	explicit Connect_job(const boost::shared_ptr<Client> &client)
		: Sync_job_base(client)
	{
		//
	}

protected:
	void really_perform(const boost::shared_ptr<Client> &client) OVERRIDE {
		POSEIDON_PROFILE_ME;

		client->on_sync_connect();
	}
};

class Client::Read_hup_job : public Client::Sync_job_base {
public:
	explicit Read_hup_job(const boost::shared_ptr<Client> &client)
		: Sync_job_base(client)
	{
		//
	}

protected:
	void really_perform(const boost::shared_ptr<Client> &client) OVERRIDE {
		POSEIDON_PROFILE_ME;

		client->shutdown_write();
	}
};

class Client::Data_message_job : public Client::Sync_job_base {
private:
	std::uint16_t m_message_id;
	Stream_buffer m_payload;

public:
	Data_message_job(const boost::shared_ptr<Client> &client, std::uint16_t message_id, Stream_buffer payload)
		: Sync_job_base(client)
		, m_message_id(message_id), m_payload(STD_MOVE(payload))
	{
		//
	}

protected:
	void really_perform(const boost::shared_ptr<Client> &client) OVERRIDE {
		POSEIDON_PROFILE_ME;

		client->on_sync_data_message(m_message_id, STD_MOVE(m_payload));
	}
};

class Client::Control_message_job : public Client::Sync_job_base {
private:
	Status_code m_status_code;
	Stream_buffer m_param;

public:
	Control_message_job(const boost::shared_ptr<Client> &client, Status_code status_code, Stream_buffer param)
		: Sync_job_base(client)
		, m_status_code(status_code), m_param(STD_MOVE(param))
	{
		//
	}

protected:
	void really_perform(const boost::shared_ptr<Client> &client) OVERRIDE {
		POSEIDON_PROFILE_ME;

		client->on_sync_control_message(m_status_code, STD_MOVE(m_param));
	}
};

Client::Client(const Sock_addr &addr, bool use_ssl, bool verify_peer)
	: Low_level_client(addr, use_ssl, verify_peer)
{
	//
}
Client::~Client(){
	//
}

void Client::on_connect(){
	POSEIDON_PROFILE_ME;

	Low_level_client::on_connect();

	Job_dispatcher::enqueue(
		boost::make_shared<Connect_job>(virtual_shared_from_this<Client>()),
		VAL_INIT);
}
void Client::on_read_hup(){
	POSEIDON_PROFILE_ME;

	Job_dispatcher::enqueue(
		boost::make_shared<Read_hup_job>(virtual_shared_from_this<Client>()),
		VAL_INIT);

	Low_level_client::on_read_hup();
}

void Client::on_low_level_data_message_header(std::uint16_t message_id, std::uint64_t /*payload_size*/){
	POSEIDON_PROFILE_ME;

	m_message_id = message_id;
	m_payload.clear();
}
void Client::on_low_level_data_message_payload(std::uint64_t /*payload_offset*/, Stream_buffer payload){
	POSEIDON_PROFILE_ME;

	m_payload.splice(payload);
}
bool Client::on_low_level_data_message_end(std::uint64_t /*payload_size*/){
	POSEIDON_PROFILE_ME;

	Job_dispatcher::enqueue(
		boost::make_shared<Data_message_job>(virtual_shared_from_this<Client>(), m_message_id, STD_MOVE(m_payload)),
		VAL_INIT);

	return true;
}

bool Client::on_low_level_control_message(Status_code status_code, Stream_buffer param){
	POSEIDON_PROFILE_ME;

	Job_dispatcher::enqueue(
		boost::make_shared<Control_message_job>(virtual_shared_from_this<Client>(), status_code, STD_MOVE(param)),
		VAL_INIT);

	return true;
}

void Client::on_sync_connect(){
	POSEIDON_PROFILE_ME;

	//
}

void Client::on_sync_control_message(Status_code status_code, Stream_buffer param){
	POSEIDON_PROFILE_ME;
	POSEIDON_LOG_TRACE("Received CBPP error message from server: status_code = ", status_code, ", param = ", param);

	if(status_code < 0){
		POSEIDON_LOG_WARNING("Received negative status code from ", get_remote_info(), ": status_code = ", status_code);
		shutdown(status_shutdown, static_cast<char *>(param.squash()));
	} else {
		switch(status_code){
		case status_shutdown:
			POSEIDON_LOG_INFO("Received SHUTDOWN frame from ", get_remote_info());
			shutdown(status_shutdown, static_cast<char *>(param.squash()));
			break;
		case status_ping:
			POSEIDON_LOG_DEBUG("Received PING frame from ", get_remote_info());
			send_control(status_pong, STD_MOVE(param));
			break;
		case status_pong:
			POSEIDON_LOG_DEBUG("Received PONG frame from ", get_remote_info());
			break;
		default:
			POSEIDON_THROW(Exception, status_unknown_control_code, Rcnts::view("Unknown control code"));
		}
	}
}

}
}
