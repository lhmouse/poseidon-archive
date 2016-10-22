// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2016, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "session.hpp"
#include "exception.hpp"
#include "utilities.hpp"
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
					"Http::Exception thrown in HTTP servlet: status_code = ", e.get_status_code(), ", what = ", e.what());
				try {
					AUTO(headers, e.get_headers());
					headers.set(sslit("Connection"), "Close");
					if(e.what()[0] == (char)0xFF){
						session->send_default(e.get_status_code(), STD_MOVE(headers));
					} else {
						session->send(e.get_status_code(), STD_MOVE(headers), StreamBuffer(e.what()));
					}
					session->shutdown_read();
					session->shutdown_write();
				} catch(...){
					session->force_shutdown();
				}
				throw;
			} catch(std::exception &e){
				LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO,
					"std::exception thrown: what = ", e.what());
				session->force_shutdown();
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

	class Session::ContinueJob : public Session::SyncJobBase {
	public:
		explicit ContinueJob(const boost::shared_ptr<Session> &session)
			: SyncJobBase(session)
		{
		}

	protected:
		void really_perform(const boost::shared_ptr<Session> &session) OVERRIDE {
			PROFILE_ME;

			session->send_default(ST_CONTINUE);
		}
	};

	class Session::RequestJob : public Session::SyncJobBase {
	private:
		RequestHeaders m_request_headers;
		std::string m_transfer_encoding;
		StreamBuffer m_entity;

	public:
		RequestJob(const boost::shared_ptr<Session> &session,
			RequestHeaders request_headers, std::string transfer_encoding, StreamBuffer entity)
			: SyncJobBase(session)
			, m_request_headers(STD_MOVE(request_headers))
			, m_transfer_encoding(STD_MOVE(transfer_encoding)), m_entity(STD_MOVE(entity))
		{
		}

	protected:
		void really_perform(const boost::shared_ptr<Session> &session) OVERRIDE {
			PROFILE_ME;

			const AUTO(keep_alive, is_keep_alive_enabled(m_request_headers));

			session->on_sync_request(STD_MOVE(m_request_headers), STD_MOVE(m_entity));

			if(keep_alive){
				const AUTO(keep_alive_timeout, MainConfig::get<boost::uint64_t>("http_keep_alive_timeout", 5000));
				session->set_timeout(keep_alive_timeout);
			} else {
				session->shutdown_read();
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

	void Session::on_read_avail(StreamBuffer data)
	try {
		PROFILE_ME;

		const AUTO(upgraded_session, get_low_level_upgraded_session());
		if(!upgraded_session){
			m_size_total += data.size();
			if(m_size_total > m_max_request_length){
				DEBUG_THROW(Exception, ST_REQUEST_ENTITY_TOO_LARGE);
			}
		}

		LowLevelSession::on_read_avail(STD_MOVE(data));
	} catch(Exception &e){
		LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO,
			"Http::Exception thrown in HTTP parser: status_code = ", e.get_status_code(), ", what = ", e.what());
		send_default(e.get_status_code());
		shutdown_read();
		shutdown_write();
	}

	void Session::on_low_level_request_headers(RequestHeaders request_headers,
		std::string transfer_encoding, boost::uint64_t content_length)
	{
		PROFILE_ME;

		(void)content_length;

		m_size_total = 0;
		m_request_headers = STD_MOVE(request_headers);
		m_transfer_encoding = STD_MOVE(transfer_encoding);
		m_entity.clear();

		const AUTO_REF(expect, m_request_headers.headers.get("Expect"));
		if(!expect.empty()){
			if(::strcasecmp(expect.c_str(), "100-continue") == 0){
				JobDispatcher::enqueue(
					boost::make_shared<ContinueJob>(virtual_shared_from_this<Session>()),
					VAL_INIT);
			} else {
				LOG_POSEIDON_DEBUG("Unknown HTTP header Expect: ", expect);
			}
		}
	}
	void Session::on_low_level_request_entity(boost::uint64_t entity_offset, bool is_chunked, StreamBuffer entity){
		PROFILE_ME;

		(void)entity_offset;
		(void)is_chunked;

		m_entity.splice(entity);
	}
	boost::shared_ptr<UpgradedSessionBase> Session::on_low_level_request_end(
		boost::uint64_t content_length, bool is_chunked, OptionalMap headers)
	{
		PROFILE_ME;

		(void)content_length;
		(void)is_chunked;

		for(AUTO(it, headers.begin()); it != headers.end(); ++it){
			m_request_headers.headers.append(it->first, STD_MOVE(it->second));
		}
		if(!is_keep_alive_enabled(m_request_headers)){
			shutdown_read();
		}

		JobDispatcher::enqueue(
			boost::make_shared<RequestJob>(
				virtual_shared_from_this<Session>(), STD_MOVE(m_request_headers), STD_MOVE(m_transfer_encoding), STD_MOVE(m_entity)),
			VAL_INIT);

		return VAL_INIT;
	}
}

}
