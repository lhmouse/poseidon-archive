// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "session.hpp"
#include "exception.hpp"
#include "utilities.hpp"
#include "../log.hpp"
#include "../profiler.hpp"
#include "../singletons/main_config.hpp"
#include "../stream_buffer.hpp"
#include "../job_base.hpp"

namespace Poseidon {

namespace Http {
	namespace {
		class SessionJobBase : public JobBase {
		private:
			const boost::weak_ptr<Session> m_session;

		protected:
			explicit SessionJobBase(const boost::shared_ptr<Session> &session)
				: m_session(session)
			{
			}

		protected:
			virtual void perform(const boost::shared_ptr<Session> &session) const = 0;

		private:
			boost::weak_ptr<const void> getCategory() const FINAL {
				return m_session;
			}
			void perform() const FINAL {
				PROFILE_ME;

				const AUTO(session, m_session.lock());
				if(!session){
					return;
				}

				try {
					perform(session);
				} catch(TryAgainLater &){
					throw;
				} catch(std::exception &e){
					LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "std::exception thrown: what = ", e.what());
					session->forceShutdown();
					throw;
				} catch(...){
					LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "Unknown exception thrown.");
					session->forceShutdown();
					throw;
				}
			}
		};
	}

	class Session::ContinueJob : public SessionJobBase {
	public:
		explicit ContinueJob(const boost::shared_ptr<Session> &session)
			: SessionJobBase(session)
		{
		}

	protected:
		void perform(const boost::shared_ptr<Session> &session) const OVERRIDE {
			PROFILE_ME;

			session->sendDefault(ST_CONTINUE);
		}
	};

	class Session::RequestJob : public SessionJobBase {
	private:
		const RequestHeaders m_requestHeaders;
		const StreamBuffer m_entity;

	public:
		RequestJob(const boost::shared_ptr<Session> &session, RequestHeaders requestHeaders, StreamBuffer entity)
			: SessionJobBase(session)
			, m_requestHeaders(STD_MOVE(requestHeaders)), m_entity(STD_MOVE(entity))
		{
		}

	protected:
		void perform(const boost::shared_ptr<Session> &session) const OVERRIDE {
			PROFILE_ME;

			try {
				LOG_POSEIDON_DEBUG("Dispatching request: URI = ", m_requestHeaders.uri);

				session->onRequest(m_requestHeaders, m_entity);

				const AUTO_REF(keepAliveStr, m_requestHeaders.headers.get("Connection"));
				if((m_requestHeaders.version < 10001)
					? (::strcasecmp(keepAliveStr.c_str(), "Keep-Alive") == 0)	// HTTP 1.0
					: (::strcasecmp(keepAliveStr.c_str(), "Close") != 0))		// HTTP 1.1
				{
					session->setTimeout(MainConfig::getConfigFile().get<boost::uint64_t>("http_keep_alive_timeout", 5000));
				} else {
					session->forceShutdown();
				}
			} catch(TryAgainLater &){
				throw;
			} catch(Exception &e){
				LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO,
					"Http::Exception thrown in HTTP servlet: URI = ", m_requestHeaders.uri, ", statusCode = ", e.statusCode());
				session->sendDefault(e.statusCode(), e.headers());
			} catch(std::exception &e){
				LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO,
					"std::exception thrown in HTTP servlet: URI = ", m_requestHeaders.uri);
				session->sendDefault(ST_BAD_REQUEST, OptionalMap());
				session->shutdownRead();
				session->shutdownWrite();
			}
		}
	};

	class Session::ErrorJob : public SessionJobBase {
	private:
		const TcpSessionBase::DelayedShutdownGuard m_guard;

		const StatusCode m_statusCode;
		const OptionalMap m_headers;

	public:
		ErrorJob(const boost::shared_ptr<Session> &session, StatusCode statusCode, OptionalMap headers)
			: SessionJobBase(session)
			, m_guard(session)
			, m_statusCode(statusCode), m_headers(STD_MOVE(headers))
		{
		}

	protected:
		void perform(const boost::shared_ptr<Session> &session) const OVERRIDE {
			PROFILE_ME;

			session->sendDefault(m_statusCode, m_headers);
		}
	};

	Session::Session(UniqueFile socket)
		: LowLevelSession(STD_MOVE(socket))
	{
	}
	Session::~Session(){
	}

	boost::shared_ptr<UpgradedLowLevelSessionBase> Session::onLowLevelRequestHeaders(
		RequestHeaders &requestHeaders, boost::uint64_t contentLength)
	{
		PROFILE_ME;

		(void)contentLength;

		const AUTO_REF(expectStr, requestHeaders.headers.get("Expect"));
		if(!expectStr.empty()){
			if(::strcasecmp(expectStr.c_str(), "100-continue") == 0){
				enqueueJob(boost::make_shared<ContinueJob>(virtualSharedFromThis<Session>()));
			} else {
				LOG_POSEIDON_WARNING("Unknown HTTP header Expect: ", expectStr);
				DEBUG_THROW(Exception, ST_BAD_REQUEST);
			}
		}

		return VAL_INIT;
	}

	void Session::onLowLevelRequest(RequestHeaders requestHeaders, StreamBuffer entity){
		PROFILE_ME;

		enqueueJob(boost::make_shared<RequestJob>(
			virtualSharedFromThis<Session>(), STD_MOVE(requestHeaders), STD_MOVE(entity)));
	}
	void Session::onLowLevelError(StatusCode statusCode, OptionalMap headers){
		PROFILE_ME;

		enqueueJob(boost::make_shared<ErrorJob>(
			virtualSharedFromThis<Session>(), statusCode, STD_MOVE(headers)));

		shutdownRead();
		shutdownWrite();
	}
}

}
