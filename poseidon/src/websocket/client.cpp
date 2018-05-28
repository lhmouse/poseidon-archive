// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "client.hpp"
#include "exception.hpp"
#include "../http/low_level_client.hpp"
#include "../option_map.hpp"
#include "../singletons/job_dispatcher.hpp"
#include "../log.hpp"
#include "../job_base.hpp"
#include "../profiler.hpp"

namespace Poseidon {
namespace Websocket {

class Client::Sync_job_base : public Job_base {
private:
	const Socket_base::Delayed_shutdown_guard m_guard;
	const boost::weak_ptr<Tcp_session_base> m_weak_parent;
	const boost::weak_ptr<Client> m_weak_client;

protected:
	explicit Sync_job_base(const boost::shared_ptr<Client> &client)
		: m_guard(boost::shared_ptr<Socket_base>(client->get_weak_parent())), m_weak_parent(client->get_weak_parent()), m_weak_client(client)
	{
		//
	}

private:
	boost::weak_ptr<const void> get_category() const FINAL {
		return m_weak_parent;
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
			POSEIDON_LOG(Logger::special_major | Logger::level_info, "Websocket::Exception thrown: status_code = ", e.get_status_code(), ", what = ", e.what());
			client->shutdown(e.get_status_code(), e.what());
		} catch(std::exception &e){
			POSEIDON_LOG(Logger::special_major | Logger::level_info, "std::exception thrown: what = ", e.what());
			client->shutdown(status_internal_error, e.what());
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
	Opcode m_opcode;
	Stream_buffer m_payload;

public:
	Data_message_job(const boost::shared_ptr<Client> &client, Opcode opcode, Stream_buffer payload)
		: Sync_job_base(client)
		, m_opcode(opcode), m_payload(STD_MOVE(payload))
	{
		//
	}

protected:
	void really_perform(const boost::shared_ptr<Client> &client) OVERRIDE {
		POSEIDON_PROFILE_ME;

		POSEIDON_LOG_DEBUG("Dispatching data message: opcode = ", m_opcode, ", payload_size = ", m_payload.size());
		client->on_sync_data_message(m_opcode, STD_MOVE(m_payload));
	}
};

class Client::Control_message_job : public Client::Sync_job_base {
private:
	Opcode m_opcode;
	Stream_buffer m_payload;

public:
	Control_message_job(const boost::shared_ptr<Client> &client, Opcode opcode, Stream_buffer payload)
		: Sync_job_base(client)
		, m_opcode(opcode), m_payload(STD_MOVE(payload))
	{
		//
	}

protected:
	void really_perform(const boost::shared_ptr<Client> &client) OVERRIDE {
		POSEIDON_PROFILE_ME;

		POSEIDON_LOG_DEBUG("Dispatching control message: opcode = ", m_opcode, ", payload_size = ", m_payload.size());
		client->on_sync_control_message(m_opcode, STD_MOVE(m_payload));
	}
};

Client::Client(const boost::shared_ptr<Http::Low_level_client> &parent)
	: Low_level_client(parent)
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

void Client::on_low_level_message_header(Opcode opcode){
	POSEIDON_PROFILE_ME;

	m_opcode = opcode;
	m_payload.clear();
}
void Client::on_low_level_message_payload(std::uint64_t /*whole_offset*/, Stream_buffer payload){
	POSEIDON_PROFILE_ME;

	m_payload.splice(payload);
}
bool Client::on_low_level_message_end(std::uint64_t /*whole_size*/){
	POSEIDON_PROFILE_ME;

	Job_dispatcher::enqueue(
		boost::make_shared<Data_message_job>(virtual_shared_from_this<Client>(), m_opcode, STD_MOVE(m_payload)),
		VAL_INIT);

	return true;
}
bool Client::on_low_level_control_message(Opcode opcode, Stream_buffer payload){
	POSEIDON_PROFILE_ME;

	Job_dispatcher::enqueue(
		boost::make_shared<Control_message_job>(virtual_shared_from_this<Client>(), opcode, STD_MOVE(payload)),
		VAL_INIT);

	return true;
}

void Client::on_sync_connect(){
	POSEIDON_PROFILE_ME;

	//
}

void Client::on_sync_control_message(Opcode opcode, Stream_buffer payload){
	POSEIDON_PROFILE_ME;
	POSEIDON_LOG_DEBUG("Control frame: opcode = ", opcode);

	const AUTO(parent, get_parent());
	if(!parent){
		return;
	}

	switch(opcode){
	case opcode_close:
		POSEIDON_LOG_INFO("Received close frame from ", parent->get_remote_info());
		shutdown(status_normal_closure, "");
		break;
	case opcode_ping:
		POSEIDON_LOG_DEBUG("Received ping frame from ", parent->get_remote_info());
		send(opcode_pong, STD_MOVE(payload));
		break;
	case opcode_pong:
		POSEIDON_LOG_DEBUG("Received pong frame from ", parent->get_remote_info());
		break;
	default:
		POSEIDON_THROW(Exception, status_protocol_error, Rcnts::view("Invalid opcode"));
	}
}

}
}
