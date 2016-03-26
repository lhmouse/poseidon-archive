// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2016, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "low_level_session.hpp"
#include "exception.hpp"
#include "control_message.hpp"
#include "../log.hpp"
#include "../profiler.hpp"
#include "../time.hpp"

namespace Poseidon {

namespace Cbpp {
	LowLevelSession::LowLevelSession(UniqueFile socket)
		: TcpSessionBase(STD_MOVE(socket))
	{
	}
	LowLevelSession::~LowLevelSession(){
	}

	void LowLevelSession::on_read_avail(StreamBuffer data){
		PROFILE_ME;

		Reader::put_encoded_data(STD_MOVE(data));
	}

	void LowLevelSession::on_data_message_header(boost::uint16_t message_id, boost::uint64_t payload_size){
		PROFILE_ME;

		on_low_level_data_message_header(message_id, payload_size);
	}
	void LowLevelSession::on_data_message_payload(boost::uint64_t payload_offset, StreamBuffer payload){
		PROFILE_ME;

		on_low_level_data_message_payload(payload_offset, STD_MOVE(payload));
	}
	bool LowLevelSession::on_data_message_end(boost::uint64_t payload_size){
		PROFILE_ME;

		return on_low_level_data_message_end(payload_size);
	}

	bool LowLevelSession::on_control_message(ControlCode control_code, boost::int64_t vint_param, std::string string_param){
		PROFILE_ME;

		return on_low_level_control_message(control_code, vint_param, STD_MOVE(string_param));
	}

	long LowLevelSession::on_encoded_data_avail(StreamBuffer encoded){
		PROFILE_ME;

		return TcpSessionBase::send(STD_MOVE(encoded));
	}

	bool LowLevelSession::send(boost::uint16_t message_id, StreamBuffer payload){
		PROFILE_ME;

		return Writer::put_data_message(message_id, STD_MOVE(payload));
	}
	bool LowLevelSession::send_error(boost::uint16_t message_id, StatusCode status_code, std::string reason){
		PROFILE_ME;

		return Writer::put_control_message(message_id, status_code, STD_MOVE(reason));
	}
}

}
