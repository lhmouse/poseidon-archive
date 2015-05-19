// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "session.hpp"
#include "exception.hpp"
#include "utilities.hpp"
#include "upgraded_session_base.hpp"
#include "../log.hpp"
#include "../profiler.hpp"
#include "../singletons/main_config.hpp"
#include "../stream_buffer.hpp"
#include "../job_base.hpp"

namespace Poseidon {

namespace Http {
	class Session::SyncJobBase : public JobBase {
	private:
		const boost::weak_ptr<Session> m_session;

	protected:
		explicit SyncJobBase(const boost::shared_ptr<Session> &session)
			: m_session(session)
		{
		}

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
			} catch(Exception &e){
				LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO,
					"Http::Exception thrown in HTTP servlet: statusCode = ", e.statusCode());
				try {
					AUTO(headers, e.headers());
					headers.set("Connection", "Close");
					session->sendDefault(e.statusCode(), STD_MOVE(headers));
					session->shutdownRead();
					session->shutdownWrite();
				} catch(...){
					session->forceShutdown();
				}
				throw;
			} catch(std::exception &e){
				LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO,
					"std::exception thrown: what = ", e.what());
				session->forceShutdown();
				throw;
			} catch(...){
				LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO,
					"Unknown exception thrown.");
				session->forceShutdown();
				throw;
			}
		}

	protected:
		virtual void perform(const boost::shared_ptr<Session> &session) const = 0;
	};

	class Session::ContinueJob : public SyncJobBase {
	public:
		explicit ContinueJob(const boost::shared_ptr<Session> &session)
			: SyncJobBase(session)
		{
		}

	protected:
		void perform(const boost::shared_ptr<Session> &session) const OVERRIDE {
			PROFILE_ME;

			session->sendDefault(ST_CONTINUE);
		}
	};

	class Session::RequestJob : public SyncJobBase {
	private:
		const RequestHeaders m_requestHeaders;
		const std::string m_transferEncoding;
		const StreamBuffer m_entity;

	public:
		RequestJob(const boost::shared_ptr<Session> &session,
			RequestHeaders requestHeaders, std::string transferEncoding, StreamBuffer entity)
			: SyncJobBase(session)
			, m_requestHeaders(STD_MOVE(requestHeaders))
			, m_transferEncoding(STD_MOVE(transferEncoding)), m_entity(STD_MOVE(entity))
		{
		}

	protected:
		void perform(const boost::shared_ptr<Session> &session) const OVERRIDE {
			PROFILE_ME;

			session->onSyncRequest(m_requestHeaders, m_entity);

			const AUTO_REF(connection, m_requestHeaders.headers.get("Connection"));
			bool keepAlive;
			if(m_requestHeaders.version < 10001){
				keepAlive = (::strcasecmp(connection.c_str(), "Keep-Alive") == 0);
			} else {
				keepAlive = (::strcasecmp(connection.c_str(), "Close") != 0);
			}
			if(keepAlive){
				session->setTimeout(MainConfig::get().get<boost::uint64_t>("http_keep_alive_timeout", 5000));
			} else {
				session->forceShutdown();
			}
		}
	};

	class Session::ErrorJob : public SyncJobBase {
	private:
		const TcpSessionBase::DelayedShutdownGuard m_guard;

		const StatusCode m_statusCode;
		const OptionalMap m_headers;

	public:
		ErrorJob(const boost::shared_ptr<Session> &session, StatusCode statusCode, OptionalMap headers)
			: SyncJobBase(session)
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
		: TcpSessionBase(STD_MOVE(socket))
		, m_sizeTotal(0), m_requestHeaders()
	{
	}
	Session::~Session(){
	}

	void Session::onReadHup() NOEXCEPT {
		PROFILE_ME;

		const AUTO(upgradedSession, m_upgradedSession);
		if(upgradedSession){
			upgradedSession->onReadHup();
		}

		TcpSessionBase::onReadHup();
	}
	void Session::onWriteHup() NOEXCEPT {
		PROFILE_ME;

		const AUTO(upgradedSession, m_upgradedSession);
		if(upgradedSession){
			upgradedSession->onWriteHup();
		}

		TcpSessionBase::onWriteHup();
	}
	void Session::onClose(int errCode) NOEXCEPT {
		PROFILE_ME;

		const AUTO(upgradedSession, m_upgradedSession);
		if(upgradedSession){
			upgradedSession->onClose(errCode);
		}

		TcpSessionBase::onClose(errCode);
	}

	void Session::onReadAvail(const void *data, std::size_t size){
		PROFILE_ME;

		// epoll 线程访问不需要锁。
		AUTO(upgradedSession, m_upgradedSession);
		if(upgradedSession){
			upgradedSession->onReadAvail(data, size);
			return;
		}

		try {
			const AUTO(maxRequestLength, MainConfig::get().get<boost::uint64_t>("http_max_request_length", 16384));
			if(m_sizeTotal > maxRequestLength){
				DEBUG_THROW(Exception, ST_REQUEST_ENTITY_TOO_LARGE);
			}

			ServerReader::putEncodedData(StreamBuffer(data, size));
		} catch(Exception &e){
			LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO,
				"Http::Exception thrown in HTTP parser: statusCode = ", e.statusCode());
			enqueueJob(boost::make_shared<ErrorJob>(
				virtualSharedFromThis<Session>(), e.statusCode(), e.headers()));
			shutdownRead();
			shutdownWrite();
			return;
		}

		upgradedSession = m_upgradedSession;
		if(upgradedSession){
			AUTO_REF(queue, ServerReader::getQueue());
			const AUTO(queueSize, queue.size());
			if(queueSize != 0){
				boost::scoped_array<char> temp(new char[queueSize]);
				queue.get(temp.get(), queueSize);
				upgradedSession->onReadAvail(temp.get(), queueSize);
			}
		}
	}

	void Session::onRequestHeaders(RequestHeaders requestHeaders, std::string transferEncoding, boost::uint64_t /* contentLength */){
		PROFILE_ME;

		const AUTO_REF(expect, requestHeaders.headers.get("Expect"));
		if(!expect.empty()){
			if(::strcasecmp(expect.c_str(), "100-continue") == 0){
				enqueueJob(boost::make_shared<ContinueJob>(virtualSharedFromThis<Session>()));
			} else {
				LOG_POSEIDON_DEBUG("Unknown HTTP header Expect: ", expect);
			}
		}

		m_requestHeaders = STD_MOVE(requestHeaders);
		m_transferEncoding = STD_MOVE(transferEncoding);
		m_entity.clear();
	}
	void Session::onRequestEntity(boost::uint64_t /* entityOffset */, StreamBuffer entity){
		PROFILE_ME;

		m_entity.splice(entity);
	}
	bool Session::onRequestEnd(boost::uint64_t /* contentLength */, OptionalMap headers){
		PROFILE_ME;

		for(AUTO(it, headers.begin()); it != headers.end(); ++it){
			m_requestHeaders.headers.append(it->first, STD_MOVE(it->second));
		}

		AUTO(upgradedSession, predispatchRequest(m_requestHeaders, m_entity));
		if(upgradedSession){
			// epoll 线程访问不需要锁。
			m_upgradedSession = STD_MOVE(upgradedSession);
			return false;
		}

		enqueueJob(boost::make_shared<RequestJob>(
			virtualSharedFromThis<Session>(), STD_MOVE(m_requestHeaders), STD_MOVE(m_transferEncoding), STD_MOVE(m_entity)));
		m_sizeTotal = 0;

		return true;
	}

	long Session::onEncodedDataAvail(StreamBuffer encoded){
		PROFILE_ME;

		return TcpSessionBase::send(STD_MOVE(encoded));
	}

	boost::shared_ptr<UpgradedSessionBase> Session::predispatchRequest(
		RequestHeaders & /* requestHeaders */, StreamBuffer & /* entity */)
	{
		PROFILE_ME;

		return VAL_INIT;
	}

	boost::shared_ptr<UpgradedSessionBase> Session::getUpgradedSession() const {
		const Poseidon::Mutex::UniqueLock lock(m_upgradedSessionMutex);
		return m_upgradedSession;
	}

	bool Session::send(ResponseHeaders responseHeaders, StreamBuffer entity){
		PROFILE_ME;

		return ServerWriter::putResponse(STD_MOVE(responseHeaders), STD_MOVE(entity));
	}
	bool Session::send(StatusCode statusCode, OptionalMap headers, StreamBuffer entity){
		PROFILE_ME;

		ResponseHeaders responseHeaders;
		responseHeaders.version = 10001;
		responseHeaders.statusCode = statusCode;
		responseHeaders.reason = getStatusCodeDesc(statusCode).descShort;
		responseHeaders.headers = STD_MOVE(headers);
		return send(STD_MOVE(responseHeaders), STD_MOVE(entity));
	}
	bool Session::sendDefault(StatusCode statusCode, OptionalMap headers){
		PROFILE_ME;

		ResponseHeaders responseHeaders;
		responseHeaders.version = 10001;
		responseHeaders.statusCode = statusCode;
		responseHeaders.reason = getStatusCodeDesc(statusCode).descShort;
		responseHeaders.headers = STD_MOVE(headers);
		return ServerWriter::putDefaultResponse(STD_MOVE(responseHeaders));
	}
}

}
