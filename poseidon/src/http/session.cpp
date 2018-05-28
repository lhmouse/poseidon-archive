// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "session.hpp"
#include "exception.hpp"
#include "../log.hpp"
#include "../profiler.hpp"
#include "../singletons/main_config.hpp"
#include "../singletons/job_dispatcher.hpp"
#include "../stream_buffer.hpp"
#include "../job_base.hpp"
#include "../atomic.hpp"

namespace Poseidon {
namespace Http {

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
			POSEIDON_LOG(Logger::special_major | Logger::level_info, "Http::Exception thrown: status_code = ", e.get_status_code(), ", what = ", e.what());
			session->send_default_and_shutdown(e.get_status_code(), e.get_headers());
		} catch(std::exception &e){
			POSEIDON_LOG(Logger::special_major | Logger::level_info, "std::exception thrown: what = ", e.what());
			session->send_default_and_shutdown(status_internal_server_error);
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

class Session::Expect_job : public Session::Sync_job_base {
private:
	Request_headers m_request_headers;

public:
	Expect_job(const boost::shared_ptr<Session> &session, Request_headers request_headers)
		: Sync_job_base(session)
		, m_request_headers(STD_MOVE(request_headers))
	{
		//
	}

protected:
	void really_perform(const boost::shared_ptr<Session> &session) OVERRIDE {
		POSEIDON_PROFILE_ME;

		session->on_sync_expect(STD_MOVE(m_request_headers));
	}
};

class Session::Request_job : public Session::Sync_job_base {
private:
	Request_headers m_request_headers;
	Stream_buffer m_entity;
	bool m_keep_alive;

public:
	Request_job(const boost::shared_ptr<Session> &session, Request_headers request_headers, Stream_buffer entity, bool keep_alive)
		: Sync_job_base(session)
		, m_request_headers(STD_MOVE(request_headers)), m_entity(STD_MOVE(entity)), m_keep_alive(keep_alive)
	{
		//
	}

protected:
	void really_perform(const boost::shared_ptr<Session> &session) OVERRIDE {
		POSEIDON_PROFILE_ME;

		session->on_sync_request(STD_MOVE(m_request_headers), STD_MOVE(m_entity));

		if(m_keep_alive){
			const AUTO(keep_alive_timeout, Main_config::get<std::uint64_t>("http_keep_alive_timeout", 5000));
			session->set_timeout(keep_alive_timeout);
		} else {
			session->shutdown_write();
		}
	}
};

Session::Session(Move<Unique_file> socket)
	: Low_level_session(STD_MOVE(socket))
	, m_max_request_length(Main_config::get<std::uint64_t>("http_max_request_length", 16384))
	, m_size_total(0), m_request_headers()
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

void Session::on_low_level_request_headers(Request_headers request_headers, std::uint64_t /*content_length*/){
	POSEIDON_PROFILE_ME;

	m_size_total = 0;
	m_request_headers = STD_MOVE(request_headers);
	m_entity.clear();

	const AUTO_REF(expect, m_request_headers.headers.get("Expect"));
	if(!expect.empty()){
		Job_dispatcher::enqueue(
			boost::make_shared<Expect_job>(virtual_shared_from_this<Session>(), m_request_headers),
			VAL_INIT);
	}
}
void Session::on_low_level_request_entity(std::uint64_t /*entity_offset*/, Stream_buffer entity){
	POSEIDON_PROFILE_ME;

	m_size_total += entity.size();
	POSEIDON_THROW_UNLESS(m_size_total <= get_max_request_length(), Exception, status_payload_too_large);
	m_entity.splice(entity);
}
boost::shared_ptr<Upgraded_session_base> Session::on_low_level_request_end(std::uint64_t content_length, Option_map headers){
	POSEIDON_PROFILE_ME;

	(void)content_length;

	for(AUTO(it, headers.begin()); it != headers.end(); ++it){
		m_request_headers.headers.append(it->first, STD_MOVE(it->second));
	}
	const bool keep_alive = is_keep_alive_enabled(m_request_headers);

	Job_dispatcher::enqueue(
		boost::make_shared<Request_job>(virtual_shared_from_this<Session>(), STD_MOVE(m_request_headers), STD_MOVE(m_entity), keep_alive),
		VAL_INIT);

	if(!keep_alive){
		shutdown_read();
	}
	return VAL_INIT;
}

void Session::on_sync_expect(Request_headers request_headers){
	POSEIDON_PROFILE_ME;

	const AUTO_REF(expect, request_headers.headers.get("Expect"));
	if(::strcasecmp(expect.c_str(), "100-continue") == 0){
		const AUTO_REF(content_length_str, request_headers.headers.get("Content-Length"));
		POSEIDON_THROW_UNLESS(!content_length_str.empty(), Exception, status_length_required);
		char *eptr;
		const AUTO(content_length, std::strtoull(content_length_str.c_str(), &eptr, 10));
		POSEIDON_THROW_UNLESS(*eptr == 0, Exception, status_bad_request);
		POSEIDON_THROW_UNLESS(content_length <= get_max_request_length(), Exception, status_payload_too_large);
		send_default(status_continue);
	} else {
		POSEIDON_LOG_WARNING("Unknown HTTP header Expect: ", expect);
		POSEIDON_THROW(Exception, status_expectation_failed);
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
