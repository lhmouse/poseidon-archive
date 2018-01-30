// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "low_level_client.hpp"
#include "exception.hpp"
#include "../http/low_level_client.hpp"
#include "../log.hpp"
#include "../profiler.hpp"

namespace Poseidon {
namespace WebSocket {

LowLevelClient::LowLevelClient(const boost::shared_ptr<Http::LowLevelClient> &parent)
	: Http::UpgradedSessionBase(parent), Reader(false), Writer()
{
	//
}
LowLevelClient::~LowLevelClient(){
	//
}

void LowLevelClient::on_connect(){
	PROFILE_ME;

	//
}
void LowLevelClient::on_read_hup(){
	PROFILE_ME;

	//
}
void LowLevelClient::on_close(int err_code){
	PROFILE_ME;

	(void)err_code;
}
void LowLevelClient::on_receive(StreamBuffer data){
	PROFILE_ME;

	Reader::put_encoded_data(STD_MOVE(data));
}

void LowLevelClient::on_data_message_header(OpCode opcode){
	PROFILE_ME;

	on_low_level_message_header(opcode);
}
void LowLevelClient::on_data_message_payload(boost::uint64_t whole_offset, StreamBuffer payload){
	PROFILE_ME;

	on_low_level_message_payload(whole_offset, STD_MOVE(payload));
}
bool LowLevelClient::on_data_message_end(boost::uint64_t whole_size){
	PROFILE_ME;

	return on_low_level_message_end(whole_size);
}

bool LowLevelClient::on_control_message(OpCode opcode, StreamBuffer payload){
	PROFILE_ME;

	return on_low_level_control_message(opcode, STD_MOVE(payload));
}

long LowLevelClient::on_encoded_data_avail(StreamBuffer encoded){
	PROFILE_ME;

	return UpgradedSessionBase::send(STD_MOVE(encoded));
}

bool LowLevelClient::send(OpCode opcode, StreamBuffer payload, bool masked){
	PROFILE_ME;

	return Writer::put_message(opcode, masked, STD_MOVE(payload));
}

bool LowLevelClient::shutdown(StatusCode status_code, const char *reason) NOEXCEPT
try {
	PROFILE_ME;

	if(has_been_shutdown_write()){
		return false;
	}
	Writer::put_close_message(status_code, true, StreamBuffer(reason));
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
