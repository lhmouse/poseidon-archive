// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "low_level_session.hpp"
#include "exception.hpp"
#include "../http/low_level_session.hpp"
#include "../log.hpp"
#include "../profiler.hpp"

namespace Poseidon {
namespace Websocket {

Low_level_session::Low_level_session(const boost::shared_ptr<Http::Low_level_session> &parent)
	: Http::Upgraded_session_base(parent), Reader(true), Writer()
{
	//
}
Low_level_session::~Low_level_session(){
	//
}

void Low_level_session::on_connect(){
	//
}
void Low_level_session::on_read_hup(){
	//
}
void Low_level_session::on_close(int /*err_code*/){
	//
}
void Low_level_session::on_receive(Stream_buffer data){
	POSEIDON_PROFILE_ME;

	Reader::put_encoded_data(STD_MOVE(data));
}

void Low_level_session::on_data_message_header(Opcode opcode){
	POSEIDON_PROFILE_ME;

	on_low_level_message_header(opcode);
}
void Low_level_session::on_data_message_payload(std::uint64_t whole_offset, Stream_buffer payload){
	POSEIDON_PROFILE_ME;

	on_low_level_message_payload(whole_offset, STD_MOVE(payload));
}
bool Low_level_session::on_data_message_end(std::uint64_t whole_size){
	POSEIDON_PROFILE_ME;

	return on_low_level_message_end(whole_size);
}

bool Low_level_session::on_control_message(Opcode opcode, Stream_buffer payload){
	POSEIDON_PROFILE_ME;

	return on_low_level_control_message(opcode, STD_MOVE(payload));
}

long Low_level_session::on_encoded_data_avail(Stream_buffer encoded){
	POSEIDON_PROFILE_ME;

	return Upgraded_session_base::send(STD_MOVE(encoded));
}

bool Low_level_session::send(Opcode opcode, Stream_buffer payload, bool masked){
	POSEIDON_PROFILE_ME;

	return Writer::put_message(opcode, masked, STD_MOVE(payload));
}

bool Low_level_session::shutdown(Status_code status_code, const char *reason) NOEXCEPT
try {
	POSEIDON_PROFILE_ME;

	if(has_been_shutdown_write()){
		return false;
	}
	Writer::put_close_message(status_code, false, Stream_buffer(reason));
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
