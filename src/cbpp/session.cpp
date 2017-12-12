// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2017, LH_Mouse. All wrongs reserved.

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

class Session::SyncJobBase : public JobBase {
private:
	const SocketBase::DelayedShutdownGuard m_guard;
	const boost::weak_ptr<Session> m_weak_session;

protected:
	explicit SyncJobBase(const boost::shared_ptr<Session> &session)
		: m_guard(session), m_weak_session(session)
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
				"Cbpp::Exception thrown: status_code = ", e.get_status_code(), ", what = ", e.what());
			session->shutdown(e.get_status_code(), e.what());
		} catch(std::exception &e){
			LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "std::exception thrown: what = ", e.what());
			session->shutdown(ST_INTERNAL_ERROR, e.what());
		} catch(...){
			LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "Unknown exception thrown.");
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
	boost::uint16_t m_message_id;
	StreamBuffer m_payload;

public:
	DataMessageJob(const boost::shared_ptr<Session> &session,
		boost::uint16_t message_id, StreamBuffer payload)
		: SyncJobBase(session)
		, m_message_id(message_id), m_payload(STD_MOVE(payload))
	{ }

protected:
	void really_perform(const boost::shared_ptr<Session> &session) OVERRIDE {
		PROFILE_ME;

		LOG_POSEIDON_DEBUG("Dispatching message: message_id = ", m_message_id, ", payload_len = ", m_payload.size());
		session->on_sync_data_message(m_message_id, STD_MOVE(m_payload));

		const AUTO(keep_alive_timeout, MainConfig::get<boost::uint64_t>("cbpp_keep_alive_timeout", 30000));
		session->set_timeout(keep_alive_timeout);
	}
};

class Session::ControlMessageJob : public Session::SyncJobBase {
private:
	StatusCode m_status_code;
	StreamBuffer m_param;

public:
	ControlMessageJob(const boost::shared_ptr<Session> &session,
		StatusCode status_code, StreamBuffer param)
		: SyncJobBase(session)
		, m_status_code(status_code), m_param(STD_MOVE(param))
	{ }

protected:
	void really_perform(const boost::shared_ptr<Session> &session) OVERRIDE {
		PROFILE_ME;

		LOG_POSEIDON_DEBUG("Dispatching control message: status_code = ", m_status_code, ", param = ", m_param);
		session->on_sync_control_message(m_status_code, STD_MOVE(m_param));

		const AUTO(keep_alive_timeout, MainConfig::get<boost::uint64_t>("cbpp_keep_alive_timeout", 30000));
		session->set_timeout(keep_alive_timeout);
	}
};

Session::Session(Move<UniqueFile> socket)
	: LowLevelSession(STD_MOVE(socket))
	, m_max_request_length(MainConfig::get<boost::uint64_t>("cbpp_max_request_length", 16384))
	, m_size_total(0), m_message_id(0), m_payload()
{ }
Session::~Session(){ }

void Session::on_read_hup(){
	PROFILE_ME;

	JobDispatcher::enqueue(
		boost::make_shared<ReadHupJob>(virtual_shared_from_this<Session>()),
		VAL_INIT);

	LowLevelSession::on_read_hup();
}

void Session::on_low_level_data_message_header(boost::uint16_t message_id, boost::uint64_t payload_size){
	PROFILE_ME;

	(void)payload_size;

	m_size_total = 0;
	m_message_id = message_id;
	m_payload.clear();
}
void Session::on_low_level_data_message_payload(boost::uint64_t payload_offset, StreamBuffer payload){
	PROFILE_ME;

	(void)payload_offset;

	m_size_total += payload.size();
	if(m_size_total > get_max_request_length()){
		DEBUG_THROW(Exception, ST_REQUEST_TOO_LARGE);
	}

	m_payload.splice(payload);
}
bool Session::on_low_level_data_message_end(boost::uint64_t payload_size){
	PROFILE_ME;

	(void)payload_size;

	JobDispatcher::enqueue(
		boost::make_shared<DataMessageJob>(virtual_shared_from_this<Session>(),
			m_message_id, STD_MOVE(m_payload)),
		VAL_INIT);

	return true;
}

bool Session::on_low_level_control_message(StatusCode status_code, StreamBuffer param){
	PROFILE_ME;

	JobDispatcher::enqueue(
		boost::make_shared<ControlMessageJob>(virtual_shared_from_this<Session>(),
			status_code, STD_MOVE(param)),
		VAL_INIT);

	return true;
}

void Session::on_sync_control_message(StatusCode status_code, StreamBuffer param){
	PROFILE_ME;
	LOG_POSEIDON_DEBUG("Recevied control message from ", get_remote_info(), ", status_code = ", status_code, ", param = ", param);

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
			send_status(ST_PONG, STD_MOVE(param));
			break;
		case ST_PONG:
			LOG_POSEIDON_DEBUG("Received PONG frame from ", get_remote_info());
			break;
		default:
			DEBUG_THROW(Exception, ST_UNKNOWN_CONTROL_CODE, sslit("Unknown control code"));
		}
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
