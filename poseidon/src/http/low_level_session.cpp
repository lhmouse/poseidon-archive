// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "low_level_session.hpp"
#include "exception.hpp"
#include "upgraded_session_base.hpp"
#include "header_option.hpp"
#include "../log.hpp"
#include "../profiler.hpp"
#include "../stream_buffer.hpp"

namespace Poseidon {
namespace Http {

Low_level_session::Low_level_session(Move<Unique_file> socket)
	: Tcp_session_base(STD_MOVE(socket)), Server_reader(), Server_writer()
{
	//
}
Low_level_session::~Low_level_session(){
	//
}

void Low_level_session::on_connect(){
	POSEIDON_PROFILE_ME;

	//
}
void Low_level_session::on_read_hup(){
	POSEIDON_PROFILE_ME;

	// epoll 线程读取不需要锁。
	const AUTO(upgraded_session, m_upgraded_session);
	if(upgraded_session){
		upgraded_session->on_read_hup();
	}
}
void Low_level_session::on_close(int err_code){
	POSEIDON_PROFILE_ME;

	// epoll 线程读取不需要锁。
	const AUTO(upgraded_session, m_upgraded_session);
	if(upgraded_session){
		upgraded_session->on_close(err_code);
	}
}
void Low_level_session::on_receive(Stream_buffer data){
	POSEIDON_PROFILE_ME;

	// epoll 线程读取不需要锁。
	AUTO(upgraded_session, m_upgraded_session);
	if(upgraded_session){
		upgraded_session->on_receive(STD_MOVE(data));
		return;
	}

	Server_reader::put_encoded_data(STD_MOVE(data));

	upgraded_session = m_upgraded_session;
	if(upgraded_session){
		upgraded_session->on_connect();

		Stream_buffer queue;
		queue.swap(Server_reader::get_queue());
		if(!queue.empty()){
			upgraded_session->on_receive(STD_MOVE(queue));
		}
	}
}

void Low_level_session::on_shutdown_timer(std::uint64_t now){
	POSEIDON_PROFILE_ME;

	// timer 线程读取需要锁。
	const AUTO(upgraded_session, get_upgraded_session());
	if(upgraded_session){
		upgraded_session->on_shutdown_timer(now);
	}

	Tcp_session_base::on_shutdown_timer(now);
}

void Low_level_session::on_request_headers(Request_headers request_headers, std::uint64_t content_length){
	POSEIDON_PROFILE_ME;

	on_low_level_request_headers(STD_MOVE(request_headers), content_length);
}
void Low_level_session::on_request_entity(std::uint64_t entity_offset, Stream_buffer entity){
	POSEIDON_PROFILE_ME;

	on_low_level_request_entity(entity_offset, STD_MOVE(entity));
}
bool Low_level_session::on_request_end(std::uint64_t content_length, Option_map headers){
	POSEIDON_PROFILE_ME;

	AUTO(upgraded_session, on_low_level_request_end(content_length, STD_MOVE(headers)));
	if(upgraded_session){
		const std::lock_guard<std::mutex> lock(m_upgraded_session_mutex);
		m_upgraded_session = STD_MOVE(upgraded_session);
		return false;
	}
	return true;
}

long Low_level_session::on_encoded_data_avail(Stream_buffer encoded){
	POSEIDON_PROFILE_ME;

	return Tcp_session_base::send(STD_MOVE(encoded));
}

boost::shared_ptr<Upgraded_session_base> Low_level_session::get_upgraded_session() const {
	const std::lock_guard<std::mutex> lock(m_upgraded_session_mutex);
	return m_upgraded_session;
}

bool Low_level_session::send(Response_headers response_headers, Stream_buffer entity){
	POSEIDON_PROFILE_ME;

	return Server_writer::put_response(STD_MOVE(response_headers), STD_MOVE(entity), true);
}
bool Low_level_session::send(Status_code status_code){
	POSEIDON_PROFILE_ME;

	return send(status_code, Option_map(), Stream_buffer());
}
bool Low_level_session::send(Status_code status_code, Stream_buffer entity, const Header_option &content_type){
	POSEIDON_PROFILE_ME;

	Option_map headers;
	headers.set(Rcnts::view("Content-Type"), content_type.dump().dump_string());
	return send(status_code, STD_MOVE(headers), STD_MOVE(entity));
}
bool Low_level_session::send(Status_code status_code, Option_map headers, Stream_buffer entity){
	POSEIDON_PROFILE_ME;

	Response_headers response_headers;
	response_headers.version = 10001;
	response_headers.status_code = status_code;
	response_headers.reason = get_status_code_desc(status_code).desc_short;
	response_headers.headers = STD_MOVE(headers);
	return send(STD_MOVE(response_headers), STD_MOVE(entity));
}

bool Low_level_session::send_chunked_header(Response_headers response_headers){
	POSEIDON_PROFILE_ME;

	return Server_writer::put_chunked_header(STD_MOVE(response_headers));
}
bool Low_level_session::send_chunk(Stream_buffer entity){
	POSEIDON_PROFILE_ME;

	return Server_writer::put_chunk(STD_MOVE(entity));
}
bool Low_level_session::send_chunked_trailer(Option_map headers){
	POSEIDON_PROFILE_ME;

	return Server_writer::put_chunked_trailer(STD_MOVE(headers));
}

bool Low_level_session::send_default(Status_code status_code, Option_map headers){
	POSEIDON_PROFILE_ME;

	AUTO(pair, make_default_response(status_code, STD_MOVE(headers)));
	return Server_writer::put_response(pair.first, STD_MOVE(pair.second), false); // no need to adjust Content-Length.
}
bool Low_level_session::send_default_and_shutdown(Status_code status_code, const Option_map &headers) NOEXCEPT
try {
	POSEIDON_PROFILE_ME;

	AUTO(pair, make_default_response(status_code, headers));
	pair.first.headers.set(Rcnts::view("Connection"), "Close");
	Server_writer::put_response(pair.first, STD_MOVE(pair.second), false); // no need to adjust Content-Length.
	shutdown_read();
	return shutdown_write();
} catch(std::exception &e){
	POSEIDON_LOG_ERROR("std::exception thrown: what = ", e.what());
	force_shutdown();
	return false;
} catch(...){
	POSEIDON_LOG_ERROR("Unknown exception thrown.");
	force_shutdown();
	return false;
}
bool Low_level_session::send_default_and_shutdown(Status_code status_code, Move<Option_map> headers) NOEXCEPT
try {
	POSEIDON_PROFILE_ME;

	if(has_been_shutdown_write()){
		return false;
	}
	AUTO(pair, make_default_response(status_code, STD_MOVE(headers)));
	pair.first.headers.set(Rcnts::view("Connection"), "Close");
	Server_writer::put_response(pair.first, STD_MOVE(pair.second), false); // no need to adjust Content-Length.
	shutdown_read();
	return shutdown_write();
} catch(std::exception &e){
	POSEIDON_LOG_ERROR("std::exception thrown: what = ", e.what());
	force_shutdown();
	return false;
} catch(...){
	POSEIDON_LOG_ERROR("Unknown exception thrown.");
	force_shutdown();
	return false;
}

}
}
