// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "low_level_client.hpp"
#include "exception.hpp"
#include "upgraded_session_base.hpp"
#include "header_option.hpp"
#include "../log.hpp"
#include "../profiler.hpp"

namespace Poseidon {
namespace Http {

Low_level_client::Low_level_client(const Sock_addr &addr, bool use_ssl, bool verify_peer)
	: Tcp_client_base(addr, use_ssl, verify_peer), Client_reader(), Client_writer()
{
	//
}
Low_level_client::~Low_level_client(){
	//
}

void Low_level_client::on_connect(){
	POSEIDON_PROFILE_ME;

	//
}
void Low_level_client::on_read_hup(){
	POSEIDON_PROFILE_ME;

	if(Client_reader::is_content_till_eof()){
		Client_reader::terminate_content();
	}

	// epoll 线程读取不需要锁。
	const AUTO(upgraded_client, m_upgraded_client);
	if(upgraded_client){
		upgraded_client->on_read_hup();
	}
}
void Low_level_client::on_close(int err_code){
	POSEIDON_PROFILE_ME;

	// epoll 线程读取不需要锁。
	const AUTO(upgraded_client, m_upgraded_client);
	if(upgraded_client){
		upgraded_client->on_close(err_code);
	}
}
void Low_level_client::on_receive(Stream_buffer data){
	POSEIDON_PROFILE_ME;

	// epoll 线程读取不需要锁。
	AUTO(upgraded_client, m_upgraded_client);
	if(upgraded_client){
		upgraded_client->on_receive(STD_MOVE(data));
		return;
	}

	Client_reader::put_encoded_data(STD_MOVE(data));

	upgraded_client = m_upgraded_client;
	if(upgraded_client){
		upgraded_client->on_connect();

		Stream_buffer queue;
		queue.swap(Client_reader::get_queue());
		if(!queue.empty()){
			upgraded_client->on_receive(STD_MOVE(queue));
		}
	}
}

void Low_level_client::on_shutdown_timer(std::uint64_t now){
	POSEIDON_PROFILE_ME;

	// timer 线程读取需要锁。
	const AUTO(upgraded_client, get_upgraded_client());
	if(upgraded_client){
		upgraded_client->on_shutdown_timer(now);
	}

	Tcp_client_base::on_shutdown_timer(now);
}

void Low_level_client::on_response_headers(Response_headers response_headers, std::uint64_t content_length){
	POSEIDON_PROFILE_ME;

	on_low_level_response_headers(STD_MOVE(response_headers), content_length);
}
void Low_level_client::on_response_entity(std::uint64_t entity_offset, Stream_buffer entity){
	POSEIDON_PROFILE_ME;

	on_low_level_response_entity(entity_offset, STD_MOVE(entity));
}
bool Low_level_client::on_response_end(std::uint64_t content_length, Option_map headers){
	POSEIDON_PROFILE_ME;

	AUTO(upgraded_client, on_low_level_response_end(content_length, STD_MOVE(headers)));
	if(upgraded_client){
		const std::lock_guard<std::mutex> lock(m_upgraded_client_mutex);
		m_upgraded_client = STD_MOVE(upgraded_client);
		return false;
	}
	return true;
}

long Low_level_client::on_encoded_data_avail(Stream_buffer encoded){
	POSEIDON_PROFILE_ME;

	return Tcp_client_base::send(STD_MOVE(encoded));
}

boost::shared_ptr<Upgraded_session_base> Low_level_client::get_upgraded_client() const {
	const std::lock_guard<std::mutex> lock(m_upgraded_client_mutex);
	return m_upgraded_client;
}

bool Low_level_client::send(Request_headers request_headers, Stream_buffer entity){
	POSEIDON_PROFILE_ME;

	return Client_writer::put_request(STD_MOVE(request_headers), STD_MOVE(entity), true);
}
bool Low_level_client::send(Verb verb, std::string uri, Option_map get_params){
	POSEIDON_PROFILE_ME;

	return send(verb, STD_MOVE(uri), STD_MOVE(get_params), Option_map(), Stream_buffer());
}
bool Low_level_client::send(Verb verb, std::string uri, Option_map get_params, Stream_buffer entity, const Header_option &content_type){
	POSEIDON_PROFILE_ME;

	Option_map headers;
	headers.set(Rcnts::view("Content-Type"), content_type.dump().dump_string());
	return send(verb, STD_MOVE(uri), STD_MOVE(get_params), STD_MOVE(headers), STD_MOVE(entity));
}
bool Low_level_client::send(Verb verb, std::string uri, Option_map get_params, Option_map headers, Stream_buffer entity){
	POSEIDON_PROFILE_ME;

	Request_headers request_headers;
	request_headers.verb = verb;
	request_headers.uri = STD_MOVE(uri);
	request_headers.version = 10001;
	request_headers.get_params = STD_MOVE(get_params);
	request_headers.headers = STD_MOVE(headers);
	return send(STD_MOVE(request_headers), STD_MOVE(entity));
}

bool Low_level_client::send_chunked_header(Request_headers request_headers){
	POSEIDON_PROFILE_ME;

	return Client_writer::put_chunked_header(STD_MOVE(request_headers));
}
bool Low_level_client::send_chunk(Stream_buffer entity){
	POSEIDON_PROFILE_ME;

	return Client_writer::put_chunk(STD_MOVE(entity));
}
bool Low_level_client::send_chunked_trailer(Option_map headers){
	POSEIDON_PROFILE_ME;

	return Client_writer::put_chunked_trailer(STD_MOVE(headers));
}

}
}
