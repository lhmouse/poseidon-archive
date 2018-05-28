// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "low_level_session.hpp"
#include "exception.hpp"
#include "../log.hpp"
#include "../profiler.hpp"
#include "../time.hpp"

namespace Poseidon {
namespace Cbpp {

Low_level_session::Low_level_session(Move<Unique_file> socket)
	: Tcp_session_base(STD_MOVE(socket)), Reader(), Writer()
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

void Low_level_session::on_data_message_header(std::uint16_t message_id, std::uint64_t payload_size){
	POSEIDON_PROFILE_ME;

	on_low_level_data_message_header(message_id, payload_size);
}
void Low_level_session::on_data_message_payload(std::uint64_t payload_offset, Stream_buffer payload){
	POSEIDON_PROFILE_ME;

	on_low_level_data_message_payload(payload_offset, STD_MOVE(payload));
}
bool Low_level_session::on_data_message_end(std::uint64_t payload_size){
	POSEIDON_PROFILE_ME;

	return on_low_level_data_message_end(payload_size);
}

bool Low_level_session::on_control_message(Status_code status_code, Stream_buffer param){
	POSEIDON_PROFILE_ME;

	return on_low_level_control_message(status_code, STD_MOVE(param));
}

long Low_level_session::on_encoded_data_avail(Stream_buffer encoded){
	POSEIDON_PROFILE_ME;

	return Tcp_session_base::send(STD_MOVE(encoded));
}

bool Low_level_session::send(std::uint16_t message_id, Stream_buffer payload){
	POSEIDON_PROFILE_ME;

	return Writer::put_data_message(message_id, STD_MOVE(payload));
}
bool Low_level_session::send_status(Status_code status_code, Stream_buffer param){
	POSEIDON_PROFILE_ME;

	return Writer::put_control_message(status_code, STD_MOVE(param));
}
bool Low_level_session::shutdown(Status_code status_code, const char *param) NOEXCEPT
try {
	POSEIDON_PROFILE_ME;

	if(has_been_shutdown_write()){
		return false;
	}
	Writer::put_control_message(status_code, Stream_buffer(param));
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
