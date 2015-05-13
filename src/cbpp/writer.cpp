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

	long Writer::putDataMessage(boost::uint16_t messageId, StreamBuffer payload){
		PROFILE_ME;

		StreamBuffer frame;
		boost::uint16_t temp16;
		boost::uint64_t temp64;
		if(payload.size() < 0xFFFF){
			storeLe(temp16, payload.size());
			frame.put(&temp16, 2);
		} else {
			storeLe(temp16, 0xFFFF);
			frame.put(&temp16, 2);
			storeLe(temp64, payload.size());
			frame.put(&temp64, 8);
		}
		storeLe(temp16, messageId);
		frame.put(&temp16, 2);
		frame.splice(payload);
		return onEncodedDataAvail(STD_MOVE(frame));
	}

	long Writer::putControlMessage(ControlCode controlCode, boost::int64_t vintParam, std::string stringParam){
		PROFILE_ME;

		return putDataMessage(ControlMessage::ID, ControlMessage(controlCode, vintParam, stringParam));
	}
}

}
