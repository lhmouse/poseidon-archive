// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2017, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "writer.hpp"
#include "../log.hpp"
#include "../profiler.hpp"
#include "../endian.hpp"
#include "../vint64.hpp"

namespace Poseidon {

namespace {
	inline void push_vint(StreamBuffer &buffer, boost::int64_t val){
		StreamBuffer::WriteIterator wit(buffer);
		vint64_to_binary(val, wit);
	}
	inline void push_vuint(StreamBuffer &buffer, boost::uint64_t val){
		StreamBuffer::WriteIterator wit(buffer);
		vuint64_to_binary(val, wit);
	}
	inline void push_blob(StreamBuffer &buffer, StreamBuffer val){
		StreamBuffer::WriteIterator wit(buffer);
		vuint64_to_binary(val.size(), wit);
		buffer.splice(val);
	}
}

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
	long Writer::put_control_message(StatusCode status_code, StreamBuffer param){
		PROFILE_ME;

		StreamBuffer payload;
		push_vint(payload, status_code);
		push_blob(payload, STD_MOVE(param));
		return put_data_message(0, STD_MOVE(payload));
	}
}

}
