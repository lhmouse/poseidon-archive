// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2016, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "low_level_client.hpp"
#include "exception.hpp"
#include "upgraded_client_base.hpp"
#include "header_option.hpp"
#include "../log.hpp"
#include "../profiler.hpp"

namespace Poseidon {

namespace Http {
	LowLevelClient::LowLevelClient(const SockAddr &addr, bool use_ssl)
		: TcpClientBase(addr, use_ssl), ClientReader(), ClientWriter()
	{
	}
	LowLevelClient::LowLevelClient(const IpPort &addr, bool use_ssl)
		: TcpClientBase(addr, use_ssl), ClientReader(), ClientWriter()
	{
	}
	LowLevelClient::~LowLevelClient(){
	}

	void LowLevelClient::on_read_hup() NOEXCEPT {
		PROFILE_ME;

		try {
			if(ClientReader::is_content_till_eof()){
				ClientReader::terminate_content();
			}
		} catch(std::exception &e){
			LOG_POSEIDON_WARNING("std::exception thrown: what = ", e.what());
			force_shutdown();
		} catch(...){
			LOG_POSEIDON_WARNING("Unknown exception thrown");
			force_shutdown();
		}

		// epoll 线程读取不需要锁。
		const AUTO(upgraded_client, m_upgraded_client);
		if(upgraded_client){
			upgraded_client->on_read_hup();
		}

		TcpClientBase::on_read_hup();
	}

	void LowLevelClient::on_read_avail(StreamBuffer data){
		PROFILE_ME;

		// epoll 线程读取不需要锁。
		AUTO(upgraded_client, m_upgraded_client);
		if(upgraded_client){
			upgraded_client->on_read_avail(STD_MOVE(data));
			return;
		}

		ClientReader::put_encoded_data(STD_MOVE(data));

		upgraded_client = m_upgraded_client;
		if(upgraded_client){
			StreamBuffer queue;
			queue.swap(ClientReader::get_queue());
			if(!queue.empty()){
				upgraded_client->on_read_avail(STD_MOVE(queue));
			}
		}
	}

	void LowLevelClient::on_response_headers(ResponseHeaders response_headers, boost::uint64_t content_length){
		PROFILE_ME;

		on_low_level_response_headers(STD_MOVE(response_headers), content_length);
	}
	void LowLevelClient::on_response_entity(boost::uint64_t entity_offset, StreamBuffer entity){
		PROFILE_ME;

		on_low_level_response_entity(entity_offset, STD_MOVE(entity));
	}
	bool LowLevelClient::on_response_end(boost::uint64_t content_length, OptionalMap headers){
		PROFILE_ME;

		AUTO(upgraded_client, on_low_level_response_end(content_length, STD_MOVE(headers)));
		if(upgraded_client){
			const Mutex::UniqueLock lock(m_upgraded_client_mutex);
			m_upgraded_client = STD_MOVE(upgraded_client);
			return false;
		}
		return true;
	}

	long LowLevelClient::on_encoded_data_avail(StreamBuffer encoded){
		PROFILE_ME;

		return TcpClientBase::send(STD_MOVE(encoded));
	}

	bool LowLevelClient::send(RequestHeaders request_headers, StreamBuffer entity){
		PROFILE_ME;

		return ClientWriter::put_request(STD_MOVE(request_headers), STD_MOVE(entity));
	}
	bool LowLevelClient::send(Verb verb, std::string uri, OptionalMap get_params){
		PROFILE_ME;

		return send(verb, STD_MOVE(uri), STD_MOVE(get_params), OptionalMap(), StreamBuffer());
	}
	bool LowLevelClient::send(Verb verb, std::string uri, OptionalMap get_params, StreamBuffer entity, const HeaderOption &content_type){
		PROFILE_ME;

		OptionalMap headers;
		headers.set(sslit("Content-Type"), content_type.dump());
		return send(verb, STD_MOVE(uri), STD_MOVE(get_params), STD_MOVE(headers), STD_MOVE(entity));
	}
	bool LowLevelClient::send(Verb verb, std::string uri, OptionalMap get_params, OptionalMap headers, StreamBuffer entity){
		PROFILE_ME;

		RequestHeaders request_headers;
		request_headers.verb = verb;
		request_headers.uri = STD_MOVE(uri);
		request_headers.version = 10001;
		request_headers.get_params = STD_MOVE(get_params);
		request_headers.headers = STD_MOVE(headers);
		return send(STD_MOVE(request_headers), STD_MOVE(entity));
	}

	bool LowLevelClient::send_chunked_header(RequestHeaders request_headers){
		PROFILE_ME;

		return ClientWriter::put_chunked_header(STD_MOVE(request_headers));
	}
	bool LowLevelClient::send_chunk(StreamBuffer entity){
		PROFILE_ME;

		return ClientWriter::put_chunk(STD_MOVE(entity));
	}
	bool LowLevelClient::send_chunked_trailer(OptionalMap headers){
		PROFILE_ME;

		return ClientWriter::put_chunked_trailer(STD_MOVE(headers));
	}
}

}
