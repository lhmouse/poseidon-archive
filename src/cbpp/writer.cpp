// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "writer.hpp"
#include "control_message.hpp"
#include "../log.hpp"
#include "../profiler.hpp"
#include "../endian.hpp"

namespace Poseidon {

namespace Cbpp {
	Writer::Writer(){
	}
	Writer::~Writer(){
	}

	long Writer::put_data_message(boost::uint16_t message_id, StreamBuffer payload){
		PROFILE_ME;

		StreamBuffer frame;
		boost::uint16_t temp16;
		boost::uint64_t temp64;
		if(payload.size() < 0xFFFF){
			store_le(temp16, payload.size());
			frame.put(&temp16, 2);
		} else {
			store_le(temp16, 0xFFFF);
			frame.put(&temp16, 2);
			store_le(temp64, payload.size());
			frame.put(&temp64, 8);
		}
		store_le(temp16, message_id);
		frame.put(&temp16, 2);
		frame.splice(payload);
		return on_encoded_data_avail(STD_MOVE(frame));
	}

	long Writer::put_control_message(ControlCode control_code, boost::int64_t vint_param, std::string string_param){
		PROFILE_ME;

		return put_data_message(ControlMessage::ID, ControlMessage(control_code, vint_param, string_param));
	}
}

}
