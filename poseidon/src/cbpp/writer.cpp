// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "writer.hpp"
#include "../log.hpp"
#include "../profiler.hpp"
#include "../endian.hpp"

namespace Poseidon {
namespace Cbpp {

Writer::Writer(){
	//
}
Writer::~Writer(){
	//
}

long Writer::put_data_message(std::uint16_t message_id, Stream_buffer payload){
	POSEIDON_PROFILE_ME;

	Stream_buffer frame;
	std::uint16_t temp16;
	std::uint64_t temp64;
	if(payload.size() < 0xFFFF){
		store_be(temp16, static_cast<std::uint16_t>(payload.size()));
		frame.put(&temp16, 2);
	} else {
		store_be(temp16, 0xFFFF);
		frame.put(&temp16, 2);
		store_be(temp64, payload.size());
		frame.put(&temp64, 8);
	}
	store_be(temp16, message_id);
	frame.put(&temp16, 2);
	frame.splice(payload);
	return on_encoded_data_avail(STD_MOVE(frame));
}
long Writer::put_control_message(Status_code status_code, Stream_buffer param){
	POSEIDON_PROFILE_ME;

	Stream_buffer payload;
	std::uint32_t temp32;
	store_be(temp32, static_cast<std::uint32_t>(boost::numeric_cast<std::int32_t>(status_code)));
	payload.put(&temp32, 4);
	payload.splice(param);
	return put_data_message(0, STD_MOVE(payload));
}

}
}
