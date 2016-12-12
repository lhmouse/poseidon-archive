// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2016, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "session.hpp"
#include "exception.hpp"
#include "../log.hpp"
#include "../profiler.hpp"
#include "../singletons/main_config.hpp"
#include "../singletons/job_dispatcher.hpp"
#include "../stream_buffer.hpp"
#include "../job_base.hpp"

namespace Poseidon {

namespace Http {
	class Session::SyncJobBase : public JobBase {
	private:
		const TcpSessionBase::DelayedShutdownGuard m_guard;
		const boost::weak_ptr<TcpSessionBase> m_category;
		const boost::weak_ptr<Session> m_weak_session;

	protected:
		explicit SyncJobBase(const boost::shared_ptr<Session> &session)
			: m_guard(session), m_category(session), m_weak_session(session)
		{
		}

	private:
		boost::weak_ptr<const void> get_category() const FINAL {
			return m_category;
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
					"Http::Exception thrown: status_code = ", e.get_status_code(), ", what = ", e.what());
				session->send_default_and_shutdown(e.get_status_code(), e.get_headers());
				throw;
			} catch(std::exception &e){
				LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO,
					"std::exception thrown: what = ", e.what());
				session->send_default_and_shutdown(ST_INTERNAL_SERVER_ERROR);
				throw;
			} catch(...){
				LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO,
					"Unknown exception thrown.");
				session->force_shutdown();
				throw;
			}
		}

	protected:
		virtual void really_perform(const boost::shared_ptr<Session> &session) = 0;
	};

	class Session::ReadHupJob : public Session::SyncJobBase {
	public:
		explicit ReadHupJob(const boost::shared_ptr<Session> &session)
			: SyncJobBase(session)
		{
		}

	protected:
		void really_perform(const boost::shared_ptr<Session> &session) OVERRIDE {
			PROFILE_ME;

			session->shutdown_write();
		}
	};

	class Session::ExpectJob : public Session::SyncJobBase {
	private:
		std::string m_expect;

	public:
		ExpectJob(const boost::shared_ptr<Session> &session, std::string expect)
			: SyncJobBase(session)
			, m_expect(STD_MOVE(expect))
		{
		}

	protected:
		void really_perform(const boost::shared_ptr<Session> &session) OVERRIDE {
			PROFILE_ME;

			if(::strcasecmp(m_expect.c_str(), "100-continue") == 0){
				session->send_default(ST_CONTINUE);
				return;
			}
			LOG_POSEIDON_WARNING("Unknown HTTP header Expect: ", m_expect);
			DEBUG_THROW(Exception, ST_EXPECTATION_FAILED);
		}
	};

	class Session::RequestJob : public Session::SyncJobBase {
	private:
		RequestHeaders m_request_headers;
		StreamBuffer m_entity;
		bool m_keep_alive;

	public:
		RequestJob(const boost::shared_ptr<Session> &session,
			RequestHeaders request_headers, StreamBuffer entity, bool keep_alive)
			: SyncJobBase(session)
			, m_request_headers(STD_MOVE(request_headers)), m_entity(STD_MOVE(entity)), m_keep_alive(keep_alive)
		{
		}

	protected:
		void really_perform(const boost::shared_ptr<Session> &session) OVERRIDE {
			PROFILE_ME;

			session->on_sync_request(STD_MOVE(m_request_headers), STD_MOVE(m_entity));

			if(m_keep_alive){
				const AUTO(keep_alive_timeout, MainConfig::get<boost::uint64_t>("http_keep_alive_timeout", 5000));
				session->set_timeout(keep_alive_timeout);
			} else {
				session->shutdown_write();
			}
		}
	};

	Session::Session(UniqueFile socket, boost::uint64_t max_request_length)
		: LowLevelSession(STD_MOVE(socket))
		, m_max_request_length(max_request_length ? max_request_length
		                                          : MainConfig::get<boost::uint64_t>("http_max_request_length", 16384))
		, m_size_total(0), m_request_headers()
	{
	}
	Session::~Session(){
	}

	void Session::on_read_hup() NOEXCEPT
	try {
		PROFILE_ME;

		JobDispatcher::enqueue(
			boost::make_shared<ReadHupJob>(virtual_shared_from_this<Session>()),
			VAL_INIT);

		LowLevelSession::on_read_hup();
	} catch(std::exception &e){
		LOG_POSEIDON_WARNING("std::exception thrown: what = ", e.what());
		force_shutdown();
	} catch(...){
		LOG_POSEIDON_WARNING("Unknown exception thrown.");
		force_shutdown();
	}

	void Session::on_read_avail(StreamBuffer data)
	try {
		PROFILE_ME;

		const AUTO(upgraded_session, get_low_level_upgraded_session());
		if(!upgraded_session){
			m_size_total += data.size();
			if(m_size_total > m_max_request_length){
				DEBUG_THROW(Exception, ST_PAYLOAD_TOO_LARGE);
			}
		}

		LowLevelSession::on_read_avail(STD_MOVE(data));
	} catch(Exception &e){
		LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO,
			"Http::Exception thrown in HTTP parser: status_code = ", e.get_status_code(), ", what = ", e.what());
		shutdown_read();
		shutdown_write();
	} catch(std::exception &e){
		LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO,
			"std::exception thrown in HTTP parser: what = ", e.what());
		shutdown_read();
		shutdown_write();
	}

	void Session::on_low_level_request_headers(RequestHeaders request_headers, boost::uint64_t content_length){
		PROFILE_ME;

		(void)content_length;

		m_size_total = 0;
		m_request_headers = STD_MOVE(request_headers);
		m_entity.clear();

		const AUTO_REF(expect, m_request_headers.headers.get("Expect"));
		if(!expect.empty()){
			JobDispatcher::enqueue(
				boost::make_shared<ExpectJob>(virtual_shared_from_this<Session>(), expect),
				VAL_INIT);
		}
	}
	void Session::on_low_level_request_entity(boost::uint64_t entity_offset, StreamBuffer entity){
		PROFILE_ME;

		(void)entity_offset;

		m_entity.splice(entity);
	}
	boost::shared_ptr<UpgradedSessionBase> Session::on_low_level_request_end(boost::uint64_t content_length, OptionalMap headers){
		PROFILE_ME;

		(void)content_length;

		for(AUTO(it, headers.begin()); it != headers.end(); ++it){
			m_request_headers.headers.append(it->first, STD_MOVE(it->second));
		}
		const bool keep_alive = is_keep_alive_enabled(m_request_headers);

		JobDispatcher::enqueue(
			boost::make_shared<RequestJob>(virtual_shared_from_this<Session>(),
				STD_MOVE(m_request_headers), STD_MOVE(m_entity), keep_alive),
			VAL_INIT);

		if(!keep_alive){
			shutdown_read();
		}
		return VAL_INIT;
	}
}

}
