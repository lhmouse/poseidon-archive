// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2016, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "low_level_session.hpp"
#include "exception.hpp"
#include "../http/low_level_session.hpp"
#include "../log.hpp"
#include "../profiler.hpp"

namespace Poseidon {

namespace WebSocket {
	LowLevelSession::LowLevelSession(const boost::shared_ptr<Http::LowLevelSession> &parent)
		: Http::UpgradedSessionBase(parent), Reader(true), Writer()
	{
	}
	LowLevelSession::~LowLevelSession(){
	}

	void LowLevelSession::on_read_avail(StreamBuffer data){
		PROFILE_ME;

		Reader::put_encoded_data(STD_MOVE(data));
	}

	void LowLevelSession::on_data_message_header(OpCode opcode){
		PROFILE_ME;

		on_low_level_message_header(opcode);
	}
	void LowLevelSession::on_data_message_payload(boost::uint64_t whole_offset, StreamBuffer payload){
		PROFILE_ME;

		on_low_level_message_payload(whole_offset, STD_MOVE(payload));
	}
	bool LowLevelSession::on_data_message_end(boost::uint64_t whole_size){
		PROFILE_ME;

		return on_low_level_message_end(whole_size);
	}

	bool LowLevelSession::on_control_message(OpCode opcode, StreamBuffer payload){
		PROFILE_ME;

		return on_low_level_control_message(opcode, STD_MOVE(payload));
	}

	long LowLevelSession::on_encoded_data_avail(StreamBuffer encoded){
		PROFILE_ME;

		return UpgradedSessionBase::send(STD_MOVE(encoded));
	}

	bool LowLevelSession::send(OpCode opcode, StreamBuffer payload, bool masked){
		PROFILE_ME;

		return Writer::put_message(opcode, masked, STD_MOVE(payload));
	}

	bool LowLevelSession::shutdown(StatusCode status_code, const char *reason) NOEXCEPT {
		PROFILE_ME;

		const AUTO(parent, get_parent());
		if(!parent){
			return false;
		}

		try {
			Writer::put_close_message(status_code, false, StreamBuffer(reason));
			parent->shutdown_read();
			return parent->shutdown_write();
		} catch(std::exception &e){
			LOG_POSEIDON_ERROR("std::exception thrown: what = ", e.what());
			parent->force_shutdown();
			return false;
		} catch(...){
			LOG_POSEIDON_ERROR("Unknown exception thrown.");
			parent->force_shutdown();
			return false;
		}
	}
}

}
