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

long Writer::put_data_message(boost::uint16_t message_id, StreamBuffer payload){
	PROFILE_ME;

	StreamBuffer frame;
	boost::uint16_t temp16;
	boost::uint64_t temp64;
	if(payload.size() < 0xFFFF){
		store_be(temp16, static_cast<boost::uint16_t>(payload.size()));
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
long Writer::put_control_message(StatusCode status_code, StreamBuffer param){
	PROFILE_ME;

	StreamBuffer payload;
	boost::uint32_t temp32;
	store_be(temp32, static_cast<boost::uint32_t>(boost::numeric_cast<boost::int32_t>(status_code)));
	payload.put(&temp32, 4);
	payload.splice(param);
	return put_data_message(0, STD_MOVE(payload));
}

}
}
