// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "session.hpp"
#include "exception.hpp"
#include "../singletons/main_config.hpp"
#include "../singletons/job_dispatcher.hpp"
#include "../log.hpp"
#include "../profiler.hpp"
#include "../job_base.hpp"
#include "../time.hpp"
#include "../atomic.hpp"

namespace Poseidon {
namespace Cbpp {

class Session::Sync_job_base : public Job_base {
private:
	const Socket_base::Delayed_shutdown_guard m_guard;
	const boost::weak_ptr<Session> m_weak_session;

protected:
	explicit Sync_job_base(const boost::shared_ptr<Session> &session)
		: m_guard(session), m_weak_session(session)
	{
		//
	}

private:
	boost::weak_ptr<const void> get_category() const FINAL {
		return m_weak_session;
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
			POSEIDON_LOG(Logger::special_major | Logger::level_info, "Cbpp::Exception thrown: status_code = ", e.get_status_code(), ", what = ", e.what());
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
		session->send_status(status_ping, Stream_buffer(str, len));
	}
};

class Session::Data_message_job : public Session::Sync_job_base {
private:
	std::uint16_t m_message_id;
	Stream_buffer m_payload;

public:
	Data_message_job(const boost::shared_ptr<Session> &session, std::uint16_t message_id, Stream_buffer payload)
		: Sync_job_base(session)
		, m_message_id(message_id), m_payload(STD_MOVE(payload))
	{
		//
	}

protected:
	void really_perform(const boost::shared_ptr<Session> &session) OVERRIDE {
		POSEIDON_PROFILE_ME;

		POSEIDON_LOG_DEBUG("Dispatching message: message_id = ", m_message_id, ", payload_len = ", m_payload.size());
		session->on_sync_data_message(m_message_id, STD_MOVE(m_payload));

		const AUTO(keep_alive_timeout, Main_config::get<std::uint64_t>("cbpp_keep_alive_timeout", 30000));
		session->set_timeout(keep_alive_timeout);
	}
};

class Session::Control_message_job : public Session::Sync_job_base {
private:
	Status_code m_status_code;
	Stream_buffer m_param;

public:
	Control_message_job(const boost::shared_ptr<Session> &session, Status_code status_code, Stream_buffer param)
		: Sync_job_base(session)
		, m_status_code(status_code), m_param(STD_MOVE(param))
	{
		//
	}

protected:
	void really_perform(const boost::shared_ptr<Session> &session) OVERRIDE {
		POSEIDON_PROFILE_ME;

		POSEIDON_LOG_DEBUG("Dispatching control message: status_code = ", m_status_code, ", param = ", m_param);
		session->on_sync_control_message(m_status_code, STD_MOVE(m_param));

		const AUTO(keep_alive_timeout, Main_config::get<std::uint64_t>("cbpp_keep_alive_timeout", 30000));
		session->set_timeout(keep_alive_timeout);
	}
};

Session::Session(Move<Unique_file> socket)
	: Low_level_session(STD_MOVE(socket))
	, m_max_request_length(Main_config::get<std::uint64_t>("cbpp_max_request_length", 16384))
	, m_size_total(0), m_message_id(0), m_payload()
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

void Session::on_low_level_data_message_header(std::uint16_t message_id, std::uint64_t /*payload_size*/){
	POSEIDON_PROFILE_ME;

	m_size_total = 0;
	m_message_id = message_id;
	m_payload.clear();
}
void Session::on_low_level_data_message_payload(std::uint64_t /*payload_offset*/, Stream_buffer payload){
	POSEIDON_PROFILE_ME;

	m_size_total += payload.size();
	POSEIDON_THROW_UNLESS(m_size_total <= get_max_request_length(), Exception, status_request_too_large);
	m_payload.splice(payload);
}
bool Session::on_low_level_data_message_end(std::uint64_t /*payload_size*/){
	POSEIDON_PROFILE_ME;

	Job_dispatcher::enqueue(
		boost::make_shared<Data_message_job>(virtual_shared_from_this<Session>(), m_message_id, STD_MOVE(m_payload)),
		VAL_INIT);

	return true;
}

bool Session::on_low_level_control_message(Status_code status_code, Stream_buffer param){
	POSEIDON_PROFILE_ME;

	Job_dispatcher::enqueue(
		boost::make_shared<Control_message_job>(virtual_shared_from_this<Session>(), status_code, STD_MOVE(param)),
		VAL_INIT);

	return true;
}

void Session::on_sync_control_message(Status_code status_code, Stream_buffer param){
	POSEIDON_PROFILE_ME;
	POSEIDON_LOG_DEBUG("Recevied control message from ", get_remote_info(), ", status_code = ", status_code, ", param = ", param);

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
			send_status(status_pong, STD_MOVE(param));
			break;
		case status_pong:
			POSEIDON_LOG_DEBUG("Received PONG frame from ", get_remote_info());
			break;
		default:
			POSEIDON_THROW(Exception, status_unknown_control_code, Rcnts::view("Unknown control code"));
		}
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
