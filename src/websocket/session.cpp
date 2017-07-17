// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2017, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "session.hpp"
#include "exception.hpp"
#include "../http/low_level_session.hpp"
#include "../optional_map.hpp"
#include "../singletons/main_config.hpp"
#include "../singletons/job_dispatcher.hpp"
#include "../log.hpp"
#include "../job_base.hpp"
#include "../profiler.hpp"
#include "../atomic.hpp"

namespace Poseidon {

namespace WebSocket {
	class Session::SyncJobBase : public JobBase {
	private:
		const SocketBase::DelayedShutdownGuard m_guard;
		const boost::weak_ptr<Session> m_weak_session;

	protected:
		explicit SyncJobBase(const boost::shared_ptr<Session> &session)
			: m_guard(boost::shared_ptr<SocketBase>(session->get_weak_parent())), m_weak_session(session)
		{ }

	private:
		boost::weak_ptr<const void> get_category() const FINAL {
			return m_weak_session;
		}
		void perform() FINAL {
			PROFILE_ME;

			const AUTO(session, m_weak_session.lock());
			if(!session || session->has_been_shutdown_write()){
				return;
			}

			try {
				really_perform(session);
			} catch(Exception &e){
				LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO,
					"WebSocket::Exception thrown: status_code = ", e.get_status_code(), ", what = ", e.what());
				session->shutdown(e.get_status_code(), e.what());
			} catch(std::exception &e){
				LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO,
					"std::exception thrown: what = ", e.what());
				session->shutdown(ST_INTERNAL_ERROR, e.what());
			} catch(...){
				LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO,
					"Unknown exception thrown.");
				session->force_shutdown();
			}
		}

	protected:
		virtual void really_perform(const boost::shared_ptr<Session> &session) = 0;
	};

	class Session::ReadHupJob : public Session::SyncJobBase {
	public:
		explicit ReadHupJob(const boost::shared_ptr<Session> &session)
			: SyncJobBase(session)
		{ }

	protected:
		void really_perform(const boost::shared_ptr<Session> &session) OVERRIDE {
			PROFILE_ME;

			session->shutdown_write();
		}
	};

	class Session::DataMessageJob : public Session::SyncJobBase {
	private:
		OpCode m_opcode;
		StreamBuffer m_payload;

	public:
		DataMessageJob(const boost::shared_ptr<Session> &session, OpCode opcode, StreamBuffer payload)
			: SyncJobBase(session)
			, m_opcode(opcode), m_payload(STD_MOVE(payload))
		{ }

	protected:
		void really_perform(const boost::shared_ptr<Session> &session) OVERRIDE {
			PROFILE_ME;

			LOG_POSEIDON_DEBUG("Dispatching data message: opcode = ", m_opcode, ", payload_size = ", m_payload.size());
			session->on_sync_data_message(m_opcode, STD_MOVE(m_payload));

			const AUTO(keep_alive_timeout, MainConfig::get<boost::uint64_t>("websocket_keep_alive_timeout", 30000));
			session->set_timeout(keep_alive_timeout);
		}
	};

	class Session::ControlMessageJob : public Session::SyncJobBase {
	private:
		OpCode m_opcode;
		StreamBuffer m_payload;

	public:
		ControlMessageJob(const boost::shared_ptr<Session> &session, OpCode opcode, StreamBuffer payload)
			: SyncJobBase(session)
			, m_opcode(opcode), m_payload(STD_MOVE(payload))
		{ }

	protected:
		void really_perform(const boost::shared_ptr<Session> &session) OVERRIDE {
			PROFILE_ME;

			LOG_POSEIDON_DEBUG("Dispatching control message: opcode = ", m_opcode, ", payload_size = ", m_payload.size());
			session->on_sync_control_message(m_opcode, STD_MOVE(m_payload));

			const AUTO(keep_alive_timeout, MainConfig::get<boost::uint64_t>("websocket_keep_alive_timeout", 30000));
			session->set_timeout(keep_alive_timeout);
		}
	};

	Session::Session(const boost::shared_ptr<Http::LowLevelSession> &parent)
		: LowLevelSession(parent)
		, m_max_request_length(MainConfig::get<boost::uint64_t>("websocket_max_request_length", 16384))
		, m_size_total(0), m_opcode(OP_INVALID)
	{ }
	Session::~Session(){ }

	void Session::on_read_hup(){
		PROFILE_ME;

		JobDispatcher::enqueue(
			boost::make_shared<ReadHupJob>(virtual_shared_from_this<Session>()),
			VAL_INIT);

		LowLevelSession::on_read_hup();
	}

	void Session::on_low_level_message_header(OpCode opcode){
		PROFILE_ME;

		m_size_total = 0;
		m_opcode = opcode;
		m_payload.clear();
	}
	void Session::on_low_level_message_payload(boost::uint64_t whole_offset, StreamBuffer payload){
		PROFILE_ME;

		(void)whole_offset;

		m_size_total += payload.size();
		if(m_size_total > get_max_request_length()){
			DEBUG_THROW(Exception, ST_MESSAGE_TOO_LARGE, sslit("Message too large"));
		}

		m_payload.splice(payload);
	}
	bool Session::on_low_level_message_end(boost::uint64_t whole_size){
		PROFILE_ME;

		(void)whole_size;

		JobDispatcher::enqueue(
			boost::make_shared<DataMessageJob>(virtual_shared_from_this<Session>(),
				m_opcode, STD_MOVE(m_payload)),
			VAL_INIT);

		return true;
	}
	bool Session::on_low_level_control_message(OpCode opcode, StreamBuffer payload){
		PROFILE_ME;

		JobDispatcher::enqueue(
			boost::make_shared<ControlMessageJob>(virtual_shared_from_this<Session>(),
				opcode, STD_MOVE(payload)),
			VAL_INIT);

		return true;
	}

	void Session::on_sync_control_message(OpCode opcode, StreamBuffer payload){
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

	boost::uint64_t Session::get_max_request_length() const {
		return atomic_load(m_max_request_length, ATOMIC_CONSUME);
	}
	void Session::set_max_request_length(boost::uint64_t max_request_length){
		atomic_store(m_max_request_length, max_request_length, ATOMIC_RELEASE);
	}
}

}
