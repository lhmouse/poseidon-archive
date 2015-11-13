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
		bool is_keep_alive_enabled(const RequestHeaders &request_headers){
			PROFILE_ME;

			const AUTO_REF(connection, request_headers.headers.get("Connection"));
			if(request_headers.version < 10001){
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
		boost::weak_ptr<const void> get_category() const FINAL {
			return m_session;
		}
		void perform() FINAL {
			PROFILE_ME;

			const AUTO(session, m_session.lock());
			if(!session){
				return;
			}

			try {
				really_perform(session);
			} catch(Exception &e){
				LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO,
					"Http::Exception thrown in HTTP servlet: status_code = ", e.status_code(), ", what = ", e.what());
				try {
					AUTO(headers, e.headers());
					headers.set(sslit("Connection"), "Close");
					if(e.what()[0] == (char)0xFF){
						session->send_default(e.status_code(), STD_MOVE(headers));
					} else {
						session->send(e.status_code(), STD_MOVE(headers), StreamBuffer(e.what()));
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
				session->set_timeout(MainConfig::get<boost::uint64_t>("http_keep_alive_timeout", 5000));
			} else {
				session->shutdown_read();
				session->shutdown_write();
			}
		}
	};

	class Session::ErrorJob : public Session::SyncJobBase {
	private:
		StatusCode m_status_code;
		OptionalMap m_headers;
		SharedNts m_message;

	public:
		ErrorJob(const boost::shared_ptr<Session> &session,
			StatusCode status_code, OptionalMap headers, SharedNts message)
			: SyncJobBase(session)
			, m_status_code(status_code), m_headers(STD_MOVE(headers)), m_message(STD_MOVE(message))
		{
		}

	protected:
		void really_perform(const boost::shared_ptr<Session> &session) OVERRIDE {
			PROFILE_ME;

			try {
				m_headers.set(sslit("Connection"), "Close");
				if(m_message[0] == (char)0xFF){
					session->send_default(m_status_code, STD_MOVE(m_headers));
				} else {
					session->send(m_status_code, STD_MOVE(m_headers), StreamBuffer(m_message.get()));
				}
				session->shutdown_write();
			} catch(...){
				session->force_shutdown();
			}
		}
	};

	Session::Session(UniqueFile socket, boost::uint64_t max_request_length)
		: TcpSessionBase(STD_MOVE(socket))
		, m_max_request_length(max_request_length ? max_request_length : MainConfig::get<boost::uint64_t>("http_max_request_length", 16384))
		, m_size_total(0), m_request_headers()
	{
	}
	Session::~Session(){
	}

	void Session::on_read_hup() NOEXCEPT {
		PROFILE_ME;

		const AUTO(upgraded_session, m_upgraded_session);
		if(upgraded_session){
			upgraded_session->on_read_hup();
		}

		TcpSessionBase::on_read_hup();
	}
	void Session::on_close(int err_code) NOEXCEPT {
		PROFILE_ME;

		const AUTO(upgraded_session, m_upgraded_session);
		if(upgraded_session){
			upgraded_session->on_close(err_code);
		}

		TcpSessionBase::on_close(err_code);
	}

	void Session::on_read_avail(StreamBuffer data){
		PROFILE_ME;

		// epoll 线程访问不需要锁。
		AUTO(upgraded_session, m_upgraded_session);
		if(upgraded_session){
			upgraded_session->on_read_avail(STD_MOVE(data));
			return;
		}

		try {
			m_size_total += data.size();
			if(m_size_total > m_max_request_length){
				DEBUG_THROW(Exception, ST_REQUEST_ENTITY_TOO_LARGE);
			}

			ServerReader::put_encoded_data(STD_MOVE(data));
		} catch(Exception &e){
			LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO,
				"Http::Exception thrown in HTTP parser: status_code = ", e.status_code(), ", what = ", e.what());
			enqueue_job(boost::make_shared<ErrorJob>(
				virtual_shared_from_this<Session>(), e.status_code(), e.headers(), SharedNts(e.what())));
			shutdown_read();
			shutdown_write();
			return;
		}

		upgraded_session = m_upgraded_session;
		if(upgraded_session){
			StreamBuffer queue;
			queue.swap(ServerReader::get_queue());
			if(!queue.empty()){
				upgraded_session->on_read_avail(STD_MOVE(queue));
			}
		}
	}

	void Session::on_request_headers(RequestHeaders request_headers, std::string transfer_encoding, boost::uint64_t /* content_length */){
		PROFILE_ME;

		const AUTO_REF(expect, request_headers.headers.get("Expect"));
		if(!expect.empty()){
			if(::strcasecmp(expect.c_str(), "100-continue") == 0){
				enqueue_job(boost::make_shared<ContinueJob>(virtual_shared_from_this<Session>()));
			} else {
				LOG_POSEIDON_DEBUG("Unknown HTTP header Expect: ", expect);
			}
		}

		m_request_headers = STD_MOVE(request_headers);
		m_transfer_encoding = STD_MOVE(transfer_encoding);
		m_entity.clear();
	}
	void Session::on_request_entity(boost::uint64_t /* entity_offset */, bool /* is_chunked */, StreamBuffer entity){
		PROFILE_ME;

		m_entity.splice(entity);
	}
	bool Session::on_request_end(boost::uint64_t /* content_length */, bool /* is_chunked */, OptionalMap headers){
		PROFILE_ME;

		for(AUTO(it, headers.begin()); it != headers.end(); ++it){
			m_request_headers.headers.append(it->first, STD_MOVE(it->second));
		}

		AUTO(upgraded_session, predispatch_request(m_request_headers, m_entity));
		if(upgraded_session){
			// epoll 线程访问不需要锁。
			m_upgraded_session = STD_MOVE(upgraded_session);
			return false;
		}

		if(!is_keep_alive_enabled(m_request_headers)){
			shutdown_read();
		}

		enqueue_job(boost::make_shared<RequestJob>(
			virtual_shared_from_this<Session>(), STD_MOVE(m_request_headers), STD_MOVE(m_transfer_encoding), STD_MOVE(m_entity)));
		m_size_total = 0;

		return true;
	}

	long Session::on_encoded_data_avail(StreamBuffer encoded){
		PROFILE_ME;

		return TcpSessionBase::send(STD_MOVE(encoded));
	}

	boost::shared_ptr<UpgradedSessionBase> Session::predispatch_request(
		RequestHeaders & /* request_headers */, StreamBuffer & /* entity */)
	{
		PROFILE_ME;

		return VAL_INIT;
	}

	boost::shared_ptr<UpgradedSessionBase> Session::get_upgraded_session() const {
		const Mutex::UniqueLock lock(m_upgraded_session_mutex);
		return m_upgraded_session;
	}

	bool Session::send(ResponseHeaders response_headers, StreamBuffer entity){
		PROFILE_ME;

		return ServerWriter::put_response(STD_MOVE(response_headers), STD_MOVE(entity));
	}
	bool Session::send(StatusCode status_code, StreamBuffer entity, std::string content_type){
		PROFILE_ME;

		OptionalMap headers;
		if(!entity.empty()){
			headers.set(sslit("Content-Type"), STD_MOVE(content_type));
		}
		return send(status_code, STD_MOVE(headers), STD_MOVE(entity));
	}
	bool Session::send(StatusCode status_code, OptionalMap headers, StreamBuffer entity){
		PROFILE_ME;

		ResponseHeaders response_headers;
		response_headers.version = 10001;
		response_headers.status_code = status_code;
		response_headers.reason = get_status_code_desc(status_code).desc_short;
		response_headers.headers = STD_MOVE(headers);
		return send(STD_MOVE(response_headers), STD_MOVE(entity));
	}
	bool Session::send_default(StatusCode status_code, OptionalMap headers){
		PROFILE_ME;

		ResponseHeaders response_headers;
		response_headers.version = 10001;
		response_headers.status_code = status_code;
		response_headers.reason = get_status_code_desc(status_code).desc_short;
		response_headers.headers = STD_MOVE(headers);
		return ServerWriter::put_default_response(STD_MOVE(response_headers));
	}
}

}
