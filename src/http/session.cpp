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
	namespace {
		bool isKeepAliveEnabled(const RequestHeaders &requestHeaders){
			PROFILE_ME;

			const AUTO_REF(connection, requestHeaders.headers.get("Connection"));
			if(requestHeaders.version < 10001){
				return ::strcasecmp(connection.c_str(), "Keep-Alive") == 0;
			} else {
				return ::strcasecmp(connection.c_str(), "Close") != 0;
			}
		}
	}

	class Session::SyncJobBase : public JobBase {
	private:
		const TcpSessionBase::DelayedShutdownGuard m_guard;
		const boost::weak_ptr<Session> m_session;

	protected:
		explicit SyncJobBase(const boost::shared_ptr<Session> &session)
			: m_guard(session), m_session(session)
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
			} catch(Exception &e){
				LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO,
					"Http::Exception thrown in HTTP servlet: statusCode = ", e.statusCode(), ", what = ", e.what());
				try {
					AUTO(headers, e.headers());
					headers.set(sslit("Connection"), "Close");
					if(e.what()[0] == (char)0xFF){
						session->sendDefault(e.statusCode(), STD_MOVE(headers));
					} else {
						session->send(e.statusCode(), STD_MOVE(headers), StreamBuffer(e.what()));
					}
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

	class Session::ContinueJob : public Session::SyncJobBase {
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

	class Session::RequestJob : public Session::SyncJobBase {
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

			if(isKeepAliveEnabled(m_requestHeaders)){
				session->setTimeout(MainConfig::get<boost::uint64_t>("http_keep_alive_timeout", 5000));
			} else {
				session->shutdownRead();
				session->shutdownWrite();
			}
		}
	};

	class Session::ErrorJob : public Session::SyncJobBase {
	private:
		mutable StatusCode m_statusCode;
		mutable OptionalMap m_headers;
		mutable SharedNts m_message;

	public:
		ErrorJob(const boost::shared_ptr<Session> &session,
			StatusCode statusCode, OptionalMap headers, SharedNts message)
			: SyncJobBase(session)
			, m_statusCode(statusCode), m_headers(STD_MOVE(headers)), m_message(STD_MOVE(message))
		{
		}

	protected:
		void perform(const boost::shared_ptr<Session> &session) const OVERRIDE {
			PROFILE_ME;

			try {
				if(m_message[0] == (char)0xFF){
					session->sendDefault(m_statusCode, STD_MOVE(m_headers));
				} else {
					session->send(m_statusCode, STD_MOVE(m_headers), StreamBuffer(m_message.get()));
				}
				session->shutdownWrite();
			} catch(...){
				session->forceShutdown();
			}
		}
	};

	Session::Session(UniqueFile socket, boost::uint64_t maxRequestLength)
		: TcpSessionBase(STD_MOVE(socket))
		, m_maxRequestLength(maxRequestLength ? maxRequestLength : MainConfig::get<boost::uint64_t>("http_max_request_length", 16384))
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
	void Session::onClose(int errCode) NOEXCEPT {
		PROFILE_ME;

		const AUTO(upgradedSession, m_upgradedSession);
		if(upgradedSession){
			upgradedSession->onClose(errCode);
		}

		TcpSessionBase::onClose(errCode);
	}

	void Session::onReadAvail(StreamBuffer data){
		PROFILE_ME;

		// epoll 线程访问不需要锁。
		AUTO(upgradedSession, m_upgradedSession);
		if(upgradedSession){
			upgradedSession->onReadAvail(STD_MOVE(data));
			return;
		}

		try {
			m_sizeTotal += data.size();
			if(m_sizeTotal > m_maxRequestLength){
				DEBUG_THROW(Exception, ST_REQUEST_ENTITY_TOO_LARGE);
			}

			ServerReader::putEncodedData(STD_MOVE(data));
		} catch(Exception &e){
			LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO,
				"Http::Exception thrown in HTTP parser: statusCode = ", e.statusCode(), ", what = ", e.what());
			enqueueJob(boost::make_shared<ErrorJob>(
				virtualSharedFromThis<Session>(), e.statusCode(), e.headers(), SharedNts(e.what())));
			shutdownRead();
			shutdownWrite();
			return;
		}

		upgradedSession = m_upgradedSession;
		if(upgradedSession){
			StreamBuffer queue;
			queue.swap(ServerReader::getQueue());
			if(!queue.empty()){
				upgradedSession->onReadAvail(STD_MOVE(queue));
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
	void Session::onRequestEntity(boost::uint64_t /* entityOffset */, bool /* isChunked */, StreamBuffer entity){
		PROFILE_ME;

		m_entity.splice(entity);
	}
	bool Session::onRequestEnd(boost::uint64_t /* contentLength */, bool /* isChunked */, OptionalMap headers){
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

		if(!isKeepAliveEnabled(m_requestHeaders)){
			shutdownRead();
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
		const Mutex::UniqueLock lock(m_upgradedSessionMutex);
		return m_upgradedSession;
	}

	bool Session::send(ResponseHeaders responseHeaders, StreamBuffer entity){
		PROFILE_ME;

		return ServerWriter::putResponse(STD_MOVE(responseHeaders), STD_MOVE(entity));
	}
	bool Session::send(StatusCode statusCode, StreamBuffer entity, std::string contentType){
		PROFILE_ME;

		OptionalMap headers;
		if(!entity.empty()){
			headers.set(sslit("Content-Type"), STD_MOVE(contentType));
		}
		return send(statusCode, STD_MOVE(headers), STD_MOVE(entity));
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
