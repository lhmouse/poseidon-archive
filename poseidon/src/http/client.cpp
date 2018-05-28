// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

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
			POSEIDON_LOG(Logger::special_major | Logger::level_info, "Http::Exception thrown in HTTP servlet: status_code = ", e.get_status_code(), ", what = ", e.what());
			client->shutdown_read();
			client->shutdown_write();
		} catch(std::exception &e){
			POSEIDON_LOG(Logger::special_major | Logger::level_info, "std::exception thrown: what = ", e.what());
			client->shutdown_read();
			client->shutdown_write();
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

class Client::Response_job : public Client::Sync_job_base {
private:
	Response_headers m_response_headers;
	Stream_buffer m_entity;

public:
	Response_job(const boost::shared_ptr<Client> &client, Response_headers response_headers, Stream_buffer entity)
		: Sync_job_base(client)
		, m_response_headers(STD_MOVE(response_headers)), m_entity(STD_MOVE(entity))
	{
		//
	}

protected:
	void really_perform(const boost::shared_ptr<Client> &client) OVERRIDE {
		POSEIDON_PROFILE_ME;

		client->on_sync_response(STD_MOVE(m_response_headers), STD_MOVE(m_entity));
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

	if(Client_reader::is_content_till_eof()){
		Client_reader::terminate_content();
	}

	Job_dispatcher::enqueue(
		boost::make_shared<Read_hup_job>(virtual_shared_from_this<Client>()),
		VAL_INIT);

	Low_level_client::on_read_hup();
}

void Client::on_low_level_response_headers(Response_headers response_headers, std::uint64_t /*content_length*/){
	POSEIDON_PROFILE_ME;

	m_response_headers = STD_MOVE(response_headers);
	m_entity.clear();
}
void Client::on_low_level_response_entity(std::uint64_t /*entity_offset*/, Stream_buffer entity){
	POSEIDON_PROFILE_ME;

	m_entity.splice(entity);
}
boost::shared_ptr<Upgraded_session_base> Client::on_low_level_response_end(std::uint64_t /*content_length*/, Option_map /*headers*/){
	POSEIDON_PROFILE_ME;

	Job_dispatcher::enqueue(
		boost::make_shared<Response_job>(virtual_shared_from_this<Client>(), STD_MOVE(m_response_headers), STD_MOVE(m_entity)),
		VAL_INIT);

	return VAL_INIT;
}

void Client::on_sync_connect(){
	POSEIDON_PROFILE_ME;

	//
}

}
}
