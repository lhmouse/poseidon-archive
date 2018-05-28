// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "writer.hpp"
#include "opcodes.hpp"
#include "../log.hpp"
#include "../profiler.hpp"
#include "../endian.hpp"
#include "../random.hpp"

namespace Poseidon {
namespace Websocket {

Writer::Writer(){
	//
}
Writer::~Writer(){
	//
}

long Writer::put_message(int opcode, bool masked, Stream_buffer payload){
	POSEIDON_PROFILE_ME;

	Stream_buffer frame;
	unsigned ch = boost::numeric_cast<unsigned>(opcode) | opmask_fin;
	frame.put(ch & 0xFF);
	const std::size_t size = payload.size();
	ch = masked ? 0x80 : 0;
	if(size < 0x7E){
		ch |= static_cast<unsigned>(size);
		frame.put(ch & 0xFF);
	} else if(size < 0x10000){
		ch |= 0x7E;
		frame.put(ch & 0xFF);
		std::uint16_t temp16;
		store_be(temp16, static_cast<std::uint16_t>(size));
		frame.put(&temp16, 2);
	} else {
		ch |= 0x7F;
		frame.put(ch & 0xFF);
		std::uint64_t temp64;
		store_be(temp64, size);
		frame.put(&temp64, 8);
	}
	if(masked){
		std::uint32_t mask = random_uint32() | 0x80808080;
		frame.put(&mask, 4);
		for(;;){
			int mb = payload.get();
			if(mb == -1){
				break;
			}
			mb ^= (int)mask;
			frame.put(mb);
			mask = (mask << 24) | (mask >> 8);
		}
	} else {
		frame.splice(payload);
	}
	return on_encoded_data_avail(STD_MOVE(frame));
}
long Writer::put_close_message(Status_code status_code, bool masked, Stream_buffer addition){
	POSEIDON_PROFILE_ME;

	Stream_buffer payload;
	std::uint16_t temp16;
	store_be(temp16, boost::numeric_cast<std::uint16_t>(status_code));
	payload.put(&temp16, 2);
	// Make sure the payload is no longer than 125 bytes, whose length fits into a byte.
	char msg[125 - 2];
	std::size_t len = addition.get(msg, sizeof(msg));
	payload.put(msg, len);
	return put_message(opcode_close, masked, STD_MOVE(payload));
}

}
}
