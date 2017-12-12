// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2017, LH_Mouse. All wrongs reserved.

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

class Client::SyncJobBase : public JobBase {
private:
	const SocketBase::DelayedShutdownGuard m_guard;
	const boost::weak_ptr<Client> m_weak_client;

protected:
	explicit SyncJobBase(const boost::shared_ptr<Client> &client)
		: m_guard(client), m_weak_client(client)
	{ }

private:
	boost::weak_ptr<const void> get_category() const FINAL {
		return m_weak_client;
	}
	void perform() FINAL {
		PROFILE_ME;

		const AUTO(client, m_weak_client.lock());
		if(!client || client->has_been_shutdown_write()){
			return;
		}

		try {
			really_perform(client);
		} catch(Exception &e){
			LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO,
				"Cbpp::Exception thrown: status_code = ", e.get_status_code(), ", what = ", e.what());
			client->force_shutdown();
		} catch(std::exception &e){
			LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO,
				"std::exception thrown: what = ", e.what());
			client->force_shutdown();
		} catch(...){
			LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO,
				"Unknown exception thrown.");
			client->force_shutdown();
		}
	}

protected:
	virtual void really_perform(const boost::shared_ptr<Client> &client) = 0;
};

class Client::ConnectJob : public Client::SyncJobBase {
public:
	explicit ConnectJob(const boost::shared_ptr<Client> &client)
		: SyncJobBase(client)
	{ }

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
	{ }

protected:
	void really_perform(const boost::shared_ptr<Client> &client) OVERRIDE {
		PROFILE_ME;

		client->shutdown_write();
	}
};

class Client::DataMessageJob : public Client::SyncJobBase {
private:
	unsigned m_message_id;
	StreamBuffer m_payload;

public:
	DataMessageJob(const boost::shared_ptr<Client> &client, unsigned message_id, StreamBuffer payload)
		: SyncJobBase(client)
		, m_message_id(message_id), m_payload(STD_MOVE(payload))
	{ }

protected:
	void really_perform(const boost::shared_ptr<Client> &client) OVERRIDE {
		PROFILE_ME;

		client->on_sync_data_message(m_message_id, STD_MOVE(m_payload));
	}
};

class Client::ControlMessageJob : public Client::SyncJobBase {
private:
	StatusCode m_status_code;
	StreamBuffer m_param;

public:
	ControlMessageJob(const boost::shared_ptr<Client> &client, StatusCode status_code, StreamBuffer param)
		: SyncJobBase(client)
		, m_status_code(status_code), m_param(STD_MOVE(param))
	{ }

protected:
	void really_perform(const boost::shared_ptr<Client> &client) OVERRIDE {
		PROFILE_ME;

		client->on_sync_control_message(m_status_code, STD_MOVE(m_param));
	}
};

Client::Client(const SockAddr &addr, bool use_ssl, bool verify_peer)
	: LowLevelClient(addr, use_ssl, verify_peer)
{ }
Client::~Client(){ }

void Client::on_connect(){
	PROFILE_ME;

	LowLevelClient::on_connect();

	JobDispatcher::enqueue(
		boost::make_shared<ConnectJob>(virtual_shared_from_this<Client>()),
		VAL_INIT);
}
void Client::on_read_hup(){
	PROFILE_ME;

	JobDispatcher::enqueue(
		boost::make_shared<ReadHupJob>(virtual_shared_from_this<Client>()),
		VAL_INIT);

	LowLevelClient::on_read_hup();
}

void Client::on_low_level_data_message_header(boost::uint16_t message_id, boost::uint64_t payload_size){
	PROFILE_ME;

	(void)payload_size;

	m_message_id = message_id;
	m_payload.clear();
}
void Client::on_low_level_data_message_payload(boost::uint64_t payload_offset, StreamBuffer payload){
	PROFILE_ME;

	(void)payload_offset;

	m_payload.splice(payload);
}
bool Client::on_low_level_data_message_end(boost::uint64_t payload_size){
	PROFILE_ME;

	(void)payload_size;

	JobDispatcher::enqueue(
		boost::make_shared<DataMessageJob>(virtual_shared_from_this<Client>(),
			m_message_id, STD_MOVE(m_payload)),
		VAL_INIT);

	return true;
}

bool Client::on_low_level_control_message(StatusCode status_code, StreamBuffer param){
	PROFILE_ME;

	JobDispatcher::enqueue(
		boost::make_shared<ControlMessageJob>(virtual_shared_from_this<Client>(),
			status_code, STD_MOVE(param)),
		VAL_INIT);

	return true;
}

void Client::on_sync_connect(){ }

void Client::on_sync_control_message(StatusCode status_code, StreamBuffer param){
	PROFILE_ME;
	LOG_POSEIDON_TRACE("Received CBPP error message from server: status_code = ", status_code, ", param = ", param);

	if(status_code < 0){
		LOG_POSEIDON_WARNING("Received negative status code from ", get_remote_info(), ": status_code = ", status_code);
		shutdown(ST_SHUTDOWN, static_cast<char *>(param.squash()));
	} else {
		switch(status_code){
		case ST_SHUTDOWN:
			LOG_POSEIDON_INFO("Received SHUTDOWN frame from ", get_remote_info());
			shutdown(ST_SHUTDOWN, static_cast<char *>(param.squash()));
			break;
		case ST_PING:
			LOG_POSEIDON_DEBUG("Received PING frame from ", get_remote_info());
			send_control(ST_PONG, STD_MOVE(param));
			break;
		case ST_PONG:
			LOG_POSEIDON_DEBUG("Received PONG frame from ", get_remote_info());
			break;
		default:
			DEBUG_THROW(Exception, ST_UNKNOWN_CONTROL_CODE, sslit("Unknown control code"));
		}
	}
}

}
}
