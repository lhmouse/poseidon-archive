// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2017, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "client.hpp"
#include "exception.hpp"
#include "../http/low_level_client.hpp"
#include "../optional_map.hpp"
#include "../singletons/main_config.hpp"
#include "../singletons/job_dispatcher.hpp"
#include "../log.hpp"
#include "../job_base.hpp"
#include "../profiler.hpp"

namespace Poseidon {

namespace {
	boost::uint64_t get_keep_alive_interval(){
		const AUTO(keep_alive_timeout, MainConfig::get<boost::uint64_t>("websocket_keep_alive_timeout", 30000));
		AUTO(keep_alive_interval, keep_alive_timeout / 2);
		if(keep_alive_interval < 1){
			keep_alive_interval = 1;
		}
		return keep_alive_interval;
	}
}

namespace WebSocket {
	class Client::SyncJobBase : public JobBase {
	private:
		const TcpClientBase::DelayedShutdownGuard m_guard;
		const boost::weak_ptr<TcpClientBase> m_category;
		const boost::weak_ptr<Client> m_weak_client;

	protected:
		explicit SyncJobBase(const boost::shared_ptr<Client> &client)
			: m_guard(client->get_safe_parent()), m_category(client->get_safe_parent()), m_weak_client(client)
		{
		}

	private:
		boost::weak_ptr<const void> get_category() const FINAL {
			return m_category;
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
					"WebSocket::Exception thrown: status_code = ", e.get_status_code(), ", what = ", e.what());
				client->shutdown(e.get_status_code(), e.what());
				throw;
			} catch(std::exception &e){
				LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO,
					"std::exception thrown: what = ", e.what());
				client->shutdown(ST_INTERNAL_ERROR, e.what());
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

	class Client::DataMessageJob : public Client::SyncJobBase {
	private:
		OpCode m_opcode;
		StreamBuffer m_payload;

	public:
		DataMessageJob(const boost::shared_ptr<Client> &client, OpCode opcode, StreamBuffer payload)
			: SyncJobBase(client)
			, m_opcode(opcode), m_payload(STD_MOVE(payload))
		{
		}

	protected:
		void really_perform(const boost::shared_ptr<Client> &client) OVERRIDE {
			PROFILE_ME;

			LOG_POSEIDON_DEBUG("Dispatching data message: opcode = ", m_opcode, ", payload_size = ", m_payload.size());
			client->on_sync_data_message(m_opcode, STD_MOVE(m_payload));
		}
	};

	class Client::ControlMessageJob : public Client::SyncJobBase {
	private:
		OpCode m_opcode;
		StreamBuffer m_payload;

	public:
		ControlMessageJob(const boost::shared_ptr<Client> &client, OpCode opcode, StreamBuffer payload)
			: SyncJobBase(client)
			, m_opcode(opcode), m_payload(STD_MOVE(payload))
		{
		}

	protected:
		void really_perform(const boost::shared_ptr<Client> &client) OVERRIDE {
			PROFILE_ME;

			LOG_POSEIDON_DEBUG("Dispatching control message: opcode = ", m_opcode, ", payload_size = ", m_payload.size());
			client->on_sync_control_message(m_opcode, STD_MOVE(m_payload));
		}
	};

	Client::Client(const boost::shared_ptr<Http::LowLevelClient> &parent)
		: LowLevelClient(parent, get_keep_alive_interval())
	{
	}
	Client::~Client(){
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

	void Client::on_low_level_message_header(OpCode opcode){
		PROFILE_ME;

		m_opcode = opcode;
		m_payload.clear();
	}
	void Client::on_low_level_message_payload(boost::uint64_t whole_offset, StreamBuffer payload){
		PROFILE_ME;

		(void)whole_offset;

		m_payload.splice(payload);
	}
	bool Client::on_low_level_message_end(boost::uint64_t whole_size){
		PROFILE_ME;

		(void)whole_size;

		JobDispatcher::enqueue(
			boost::make_shared<DataMessageJob>(virtual_shared_from_this<Client>(),
				m_opcode, STD_MOVE(m_payload)),
			VAL_INIT);

		return true;
	}
	bool Client::on_low_level_control_message(OpCode opcode, StreamBuffer payload){
		PROFILE_ME;

		JobDispatcher::enqueue(
			boost::make_shared<ControlMessageJob>(virtual_shared_from_this<Client>(),
				opcode, STD_MOVE(payload)),
			VAL_INIT);

		return true;
	}

	void Client::on_sync_control_message(OpCode opcode, StreamBuffer payload){
		PROFILE_ME;
		LOG_POSEIDON_DEBUG("Control frame: opcode = ", opcode);

		const AUTO(parent, get_parent());
		if(!parent){
			return;
		}

		switch(opcode){
		case OP_CLOSE:
			LOG_POSEIDON_INFO("Received close frame from ", parent->get_remote_info());
			shutdown(ST_NORMAL_CLOSURE, "");
			break;
		case OP_PING:
			LOG_POSEIDON_INFO("Received ping frame from ", parent->get_remote_info());
			send(OP_PONG, STD_MOVE(payload));
			break;
		case OP_PONG:
			LOG_POSEIDON_INFO("Received pong frame from ", parent->get_remote_info());
			break;
		default:
			DEBUG_THROW(Exception, ST_PROTOCOL_ERROR, sslit("Invalid opcode"));
			break;
		}
	}
}

}
