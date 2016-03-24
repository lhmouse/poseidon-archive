// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2016, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "low_level_client.hpp"
#include "exception.hpp"
#include "status_codes.hpp"
#include "../log.hpp"
#include "../profiler.hpp"

namespace Poseidon {

namespace Http {
	LowLevelClient::LowLevelClient(const SockAddr &addr, bool use_ssl)
		: TcpClientBase(addr, use_ssl)
	{
	}
	LowLevelClient::LowLevelClient(const IpPort &addr, bool use_ssl)
		: TcpClientBase(addr, use_ssl)
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

		TcpClientBase::on_read_hup();
	}

	void LowLevelClient::on_read_avail(StreamBuffer data){
		PROFILE_ME;

		ClientReader::put_encoded_data(STD_MOVE(data));
	}

	void LowLevelClient::on_response_headers(ResponseHeaders response_headers,
		std::string transfer_encoding, boost::uint64_t content_length)
	{
		PROFILE_ME;

		on_low_level_response_headers(STD_MOVE(response_headers), STD_MOVE(transfer_encoding), content_length);
	}
	void LowLevelClient::on_response_entity(boost::uint64_t entity_offset, bool is_chunked, StreamBuffer entity){
		PROFILE_ME;

		on_low_level_response_entity(entity_offset, is_chunked, STD_MOVE(entity));
	}
	bool LowLevelClient::on_response_end(boost::uint64_t content_length, bool is_chunked, OptionalMap headers){
		PROFILE_ME;

		return on_low_level_response_end(content_length, is_chunked, STD_MOVE(headers));
	}

	long LowLevelClient::on_encoded_data_avail(StreamBuffer encoded){
		PROFILE_ME;

		return TcpClientBase::send(STD_MOVE(encoded));
	}

	bool LowLevelClient::send_headers(RequestHeaders request_headers){
		PROFILE_ME;

		return ClientWriter::put_request_headers(STD_MOVE(request_headers));
	}
	bool LowLevelClient::send_entity(StreamBuffer data){
		PROFILE_ME;

		return ClientWriter::put_entity(STD_MOVE(data));
	}

	bool LowLevelClient::send(RequestHeaders request_headers, StreamBuffer entity){
		PROFILE_ME;

		return ClientWriter::put_request(STD_MOVE(request_headers), STD_MOVE(entity));
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
