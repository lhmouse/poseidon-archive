// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "session.hpp"
#include "exception.hpp"
#include "../http/low_level_session.hpp"
#include "../option_map.hpp"
#include "../singletons/main_config.hpp"
#include "../singletons/job_dispatcher.hpp"
#include "../log.hpp"
#include "../job_base.hpp"
#include "../profiler.hpp"
#include "../time.hpp"
#include "../atomic.hpp"

namespace Poseidon {
namespace Websocket {

class Session::Sync_job_base : public Job_base {
private:
	const Socket_base::Delayed_shutdown_guard m_guard;
	const boost::weak_ptr<Tcp_session_base> m_weak_parent;
	const boost::weak_ptr<Session> m_weak_session;

protected:
	explicit Sync_job_base(const boost::shared_ptr<Session> &session)
		: m_guard(boost::shared_ptr<Socket_base>(session->get_weak_parent())), m_weak_parent(session->get_weak_parent()), m_weak_session(session)
	{
		//
	}

private:
	boost::weak_ptr<const void> get_category() const FINAL {
		return m_weak_parent;
	}
	void perform() FINAL {
		POSEIDON_PROFILE_ME;

		const AUTO(session, m_weak_session.lock());
		if(!session || session->has_been_shutdown_write()){
			return;
		}

		try {
			really_perform(session);
		} catch(Exception &e){
			POSEIDON_LOG(Logger::special_major | Logger::level_info, "Websocket::Exception thrown: status_code = ", e.get_status_code(), ", what = ", e.what());
			session->shutdown(e.get_status_code(), e.what());
		} catch(std::exception &e){
			POSEIDON_LOG(Logger::special_major | Logger::level_info, "std::exception thrown: what = ", e.what());
			session->shutdown(status_internal_error, e.what());
		} catch(...){
			POSEIDON_LOG(Logger::special_major | Logger::level_info, "Unknown exception thrown.");
			session->force_shutdown();
		}
	}

protected:
	virtual void really_perform(const boost::shared_ptr<Session> &session) = 0;
};

class Session::Read_hup_job : public Session::Sync_job_base {
public:
	explicit Read_hup_job(const boost::shared_ptr<Session> &session)
		: Sync_job_base(session)
	{
		//
	}

protected:
	void really_perform(const boost::shared_ptr<Session> &session) OVERRIDE {
		POSEIDON_PROFILE_ME;

		session->shutdown_write();
	}
};

class Session::Ping_job : public Session::Sync_job_base {
public:
	explicit Ping_job(const boost::shared_ptr<Session> &session)
		: Sync_job_base(session)
	{
		//
	}

protected:
	void really_perform(const boost::shared_ptr<Session> &session) OVERRIDE {
		POSEIDON_PROFILE_ME;

		const std::uint64_t local_now = get_local_time();
		char str[256];
		std::size_t len = format_time(str, sizeof(str), local_now, true);
		session->send(opcode_ping, Stream_buffer(str, len));
	}
};

class Session::Data_message_job : public Session::Sync_job_base {
private:
	Opcode m_opcode;
	Stream_buffer m_payload;

public:
	Data_message_job(const boost::shared_ptr<Session> &session, Opcode opcode, Stream_buffer payload)
		: Sync_job_base(session)
		, m_opcode(opcode), m_payload(STD_MOVE(payload))
	{
		//
	}

protected:
	void really_perform(const boost::shared_ptr<Session> &session) OVERRIDE {
		POSEIDON_PROFILE_ME;

		POSEIDON_LOG_DEBUG("Dispatching data message: opcode = ", m_opcode, ", payload_size = ", m_payload.size());
		session->on_sync_data_message(m_opcode, STD_MOVE(m_payload));

		const AUTO(keep_alive_timeout, Main_config::get<std::uint64_t>("websocket_keep_alive_timeout", 30000));
		session->set_timeout(keep_alive_timeout);
	}
};

class Session::Control_message_job : public Session::Sync_job_base {
private:
	Opcode m_opcode;
	Stream_buffer m_payload;

public:
	Control_message_job(const boost::shared_ptr<Session> &session, Opcode opcode, Stream_buffer payload)
		: Sync_job_base(session)
		, m_opcode(opcode), m_payload(STD_MOVE(payload))
	{
		//
	}

protected:
	void really_perform(const boost::shared_ptr<Session> &session) OVERRIDE {
		POSEIDON_PROFILE_ME;

		POSEIDON_LOG_DEBUG("Dispatching control message: opcode = ", m_opcode, ", payload_size = ", m_payload.size());
		session->on_sync_control_message(m_opcode, STD_MOVE(m_payload));

		const AUTO(keep_alive_timeout, Main_config::get<std::uint64_t>("websocket_keep_alive_timeout", 30000));
		session->set_timeout(keep_alive_timeout);
	}
};

Session::Session(const boost::shared_ptr<Http::Low_level_session> &parent)
	: Low_level_session(parent)
	, m_max_request_length(Main_config::get<std::uint64_t>("websocket_max_request_length", 16384))
	, m_size_total(0), m_opcode(opcode_invalid)
{
	//
}
Session::~Session(){
	//
}

void Session::on_read_hup(){
	POSEIDON_PROFILE_ME;

	Job_dispatcher::enqueue(
		boost::make_shared<Read_hup_job>(virtual_shared_from_this<Session>()),
		VAL_INIT);

	Low_level_session::on_read_hup();
}
void Session::on_shutdown_timer(std::uint64_t now){
	POSEIDON_PROFILE_ME;

	Job_dispatcher::enqueue(
		boost::make_shared<Ping_job>(virtual_shared_from_this<Session>()),
		VAL_INIT);

	Low_level_session::on_shutdown_timer(now);
}

void Session::on_low_level_message_header(Opcode opcode){
	POSEIDON_PROFILE_ME;

	m_size_total = 0;
	m_opcode = opcode;
	m_payload.clear();
}
void Session::on_low_level_message_payload(std::uint64_t /*whole_offset*/, Stream_buffer payload){
	POSEIDON_PROFILE_ME;

	m_size_total += payload.size();
	POSEIDON_THROW_UNLESS(m_size_total <= get_max_request_length(), Exception, status_message_too_large, Rcnts::view("Message too large"));
	m_payload.splice(payload);
}
bool Session::on_low_level_message_end(std::uint64_t /*whole_size*/){
	POSEIDON_PROFILE_ME;

	Job_dispatcher::enqueue(
		boost::make_shared<Data_message_job>(virtual_shared_from_this<Session>(), m_opcode, STD_MOVE(m_payload)),
		VAL_INIT);

	return true;
}
bool Session::on_low_level_control_message(Opcode opcode, Stream_buffer payload){
	POSEIDON_PROFILE_ME;

	Job_dispatcher::enqueue(
		boost::make_shared<Control_message_job>(virtual_shared_from_this<Session>(), opcode, STD_MOVE(payload)),
		VAL_INIT);

	return true;
}

void Session::on_sync_control_message(Opcode opcode, Stream_buffer payload){
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

std::uint64_t Session::get_max_request_length() const {
	return atomic_load(m_max_request_length, memory_order_consume);
}
void Session::set_max_request_length(std::uint64_t max_request_length){
	atomic_store(m_max_request_length, max_request_length, memory_order_release);
}

}
}
