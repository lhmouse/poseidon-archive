// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "low_level_client.hpp"
#include "exception.hpp"
#include "../log.hpp"
#include "../profiler.hpp"

namespace Poseidon {
namespace Cbpp {

LowLevelClient::LowLevelClient(const SockAddr &addr, bool use_ssl, bool verify_peer)
	: TcpClientBase(addr, use_ssl, verify_peer), Reader(), Writer()
{
	//
}
LowLevelClient::~LowLevelClient(){
	//
}

void LowLevelClient::on_connect(){
	//
}
void LowLevelClient::on_read_hup(){
	//
}
void LowLevelClient::on_close(int /*err_code*/){
	//
}
void LowLevelClient::on_receive(StreamBuffer data){
	PROFILE_ME;

	Reader::put_encoded_data(STD_MOVE(data));
}

void LowLevelClient::on_data_message_header(boost::uint16_t message_id, boost::uint64_t payload_size){
	PROFILE_ME;

	on_low_level_data_message_header(message_id, payload_size);
}
void LowLevelClient::on_data_message_payload(boost::uint64_t payload_offset, StreamBuffer payload){
	PROFILE_ME;

	on_low_level_data_message_payload(payload_offset, STD_MOVE(payload));
}
bool LowLevelClient::on_data_message_end(boost::uint64_t payload_size){
	PROFILE_ME;

	return on_low_level_data_message_end(payload_size);
}

bool LowLevelClient::on_control_message(StatusCode status_code, StreamBuffer param){
	PROFILE_ME;

	return on_low_level_control_message(status_code, STD_MOVE(param));
}

long LowLevelClient::on_encoded_data_avail(StreamBuffer encoded){
	PROFILE_ME;

	return TcpClientBase::send(STD_MOVE(encoded));
}

bool LowLevelClient::send(boost::uint16_t message_id, StreamBuffer payload){
	PROFILE_ME;

	return Writer::put_data_message(message_id, STD_MOVE(payload));
}
bool LowLevelClient::send_control(StatusCode status_code, StreamBuffer param){
	PROFILE_ME;

	return Writer::put_control_message(status_code, STD_MOVE(param));
}
bool LowLevelClient::shutdown(StatusCode status_code, const char *reason) NOEXCEPT
try {
	PROFILE_ME;

	if(has_been_shutdown_write()){
		return false;
	}
	Writer::put_control_message(status_code, StreamBuffer(reason));
	shutdown_read();
	return shutdown_write();
} catch(std::exception &e){
	LOG_POSEIDON_ERROR("std::exception thrown: what = ", e.what());
	force_shutdown();
	return false;
} catch(...){
	LOG_POSEIDON_ERROR("Unknown exception thrown.");
	force_shutdown();
	return false;
}

}
}
