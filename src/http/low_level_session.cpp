// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2016, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "low_level_session.hpp"
#include "exception.hpp"
#include "utilities.hpp"
#include "upgraded_session_base.hpp"
#include "../log.hpp"
#include "../profiler.hpp"
#include "../singletons/main_config.hpp"
#include "../stream_buffer.hpp"

namespace Poseidon {

namespace Http {
	LowLevelSession::LowLevelSession(UniqueFile socket, boost::uint64_t max_request_length)
		: TcpSessionBase(STD_MOVE(socket))
		, m_max_request_length(max_request_length ? max_request_length
		                                          : MainConfig::get<boost::uint64_t>("http_max_request_length", 16384))
		, m_size_total(0), m_request_headers()
	{
	}
	LowLevelSession::~LowLevelSession(){
	}

	void LowLevelSession::on_read_hup() NOEXCEPT {
		PROFILE_ME;

		// epoll 线程访问不需要锁。
		const AUTO(upgraded_session, m_upgraded_session);
		if(upgraded_session){
			upgraded_session->on_read_hup();
		}

		TcpSessionBase::on_read_hup();
	}
	void LowLevelSession::on_close(int err_code) NOEXCEPT {
		PROFILE_ME;

		// epoll 线程访问不需要锁。
		const AUTO(upgraded_session, m_upgraded_session);
		if(upgraded_session){
			upgraded_session->on_close(err_code);
		}

		TcpSessionBase::on_close(err_code);
	}

	void LowLevelSession::on_read_avail(StreamBuffer data){
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
			force_shutdown();
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

	void LowLevelSession::on_request_headers(RequestHeaders request_headers, std::string transfer_encoding, boost::uint64_t /* content_length */){
		PROFILE_ME;

		m_request_headers = STD_MOVE(request_headers);
		m_transfer_encoding = STD_MOVE(transfer_encoding);
		m_entity.clear();
	}
	void LowLevelSession::on_request_entity(boost::uint64_t /* entity_offset */, bool /* is_chunked */, StreamBuffer entity){
		PROFILE_ME;

		m_entity.splice(entity);
	}
	bool LowLevelSession::on_request_end(boost::uint64_t /* content_length */, bool /* is_chunked */, OptionalMap headers){
		PROFILE_ME;

		AUTO(request_headers, STD_MOVE_IDN(m_request_headers));
		AUTO(transfer_encoding, STD_MOVE_IDN(m_transfer_encoding));
		AUTO(entity, STD_MOVE_IDN(m_entity));

		m_size_total = 0;
		m_request_headers = VAL_INIT;
		m_transfer_encoding.clear();
		m_entity.clear();

		for(AUTO(it, headers.begin()); it != headers.end(); ++it){
			request_headers.headers.append(it->first, STD_MOVE(it->second));
		}
		if(!is_keep_alive_enabled(request_headers)){
			shutdown_read();
		}

		AUTO(upgraded_session, on_low_level_request(STD_MOVE(request_headers), STD_MOVE(transfer_encoding), STD_MOVE(entity)));
		if(upgraded_session){
			const Mutex::UniqueLock lock(m_upgraded_session_mutex);
			m_upgraded_session = STD_MOVE(upgraded_session);
			return false;
		}
		return true;
	}

	long LowLevelSession::on_encoded_data_avail(StreamBuffer encoded){
		PROFILE_ME;

		return TcpSessionBase::send(STD_MOVE(encoded));
	}

	boost::shared_ptr<UpgradedSessionBase> LowLevelSession::get_upgraded_session() const {
		const Mutex::UniqueLock lock(m_upgraded_session_mutex);
		return m_upgraded_session;
	}

	bool LowLevelSession::send(ResponseHeaders response_headers, StreamBuffer entity){
		PROFILE_ME;

		return ServerWriter::put_response(STD_MOVE(response_headers), STD_MOVE(entity));
	}
	bool LowLevelSession::send(StatusCode status_code, StreamBuffer entity, std::string content_type){
		PROFILE_ME;

		OptionalMap headers;
		if(!entity.empty()){
			headers.set(sslit("Content-Type"), STD_MOVE(content_type));
		}
		return send(status_code, STD_MOVE(headers), STD_MOVE(entity));
	}
	bool LowLevelSession::send(StatusCode status_code, OptionalMap headers, StreamBuffer entity){
		PROFILE_ME;

		ResponseHeaders response_headers;
		response_headers.version = 10001;
		response_headers.status_code = status_code;
		response_headers.reason = get_status_code_desc(status_code).desc_short;
		response_headers.headers = STD_MOVE(headers);
		return send(STD_MOVE(response_headers), STD_MOVE(entity));
	}
	bool LowLevelSession::send_default(StatusCode status_code, OptionalMap headers){
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
