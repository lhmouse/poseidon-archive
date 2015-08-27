// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "writer.hpp"
#include "opcodes.hpp"
#include "../log.hpp"
#include "../profiler.hpp"
#include "../endian.hpp"
#include "../random.hpp"

namespace Poseidon {

namespace WebSocket {
	Writer::Writer(){
	}
	Writer::~Writer(){
	}

	long Writer::putMessage(int opcode, bool masked, StreamBuffer payload){
		PROFILE_ME;

		StreamBuffer frame;
		unsigned char ch = opcode | OP_FL_FIN;
		frame.put(ch);
		const std::size_t size = payload.size();
		ch = masked ? 0x80 : 0;
		if(size < 0x7E){
			ch |= size;
			frame.put(ch);
		} else if(size < 0x10000){
			ch |= 0x7E;
			frame.put(ch);
			boost::uint16_t temp16;
			storeBe(temp16, size);
			frame.put(&temp16, 2);
		} else {
			ch |= 0x7F;
			frame.put(ch);
			boost::uint64_t temp64;
			storeBe(temp64, size);
			frame.put(&temp64, 8);
		}
		if(masked){
			boost::uint32_t mask;
			storeLe(mask, rand32() | 0x80808080u);
			frame.put(&mask, 4);
			int ch;
			for(;;){
				ch = payload.get();
				if(ch == -1){
					break;
				}
				ch ^= static_cast<unsigned char>(mask);
				frame.put(ch);
				mask = (mask << 24) | (mask >> 8);
			}
		} else {
			frame.splice(payload);
		}
		return onEncodedDataAvail(STD_MOVE(frame));
	}
	long Writer::putCloseMessage(StatusCode statusCode, StreamBuffer additional){
		PROFILE_ME;

		StreamBuffer payload;
		boost::uint16_t temp16;
		storeBe(temp16, statusCode);
		payload.put(&temp16, 2);
		char msg[0x7B];
		unsigned len = additional.get(msg, sizeof(msg));
		payload.put(msg, len);
		return putMessage(OP_CLOSE, false, STD_MOVE(payload));
	}
}

}
