// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2016, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "low_level_session.hpp"
#include "exception.hpp"
#include "../http/low_level_session.hpp"
#include "../optional_map.hpp"
#include "../singletons/main_config.hpp"
#include "../log.hpp"
#include "../profiler.hpp"

namespace Poseidon {

namespace WebSocket {
	LowLevelSession::LowLevelSession(const boost::shared_ptr<Http::LowLevelSession> &parent, boost::uint64_t max_request_length)
		: Http::UpgradedSessionBase(parent)
		, m_max_request_length(max_request_length ? max_request_length
		                                          : MainConfig::get<boost::uint64_t>("websocket_max_request_length", 16384))
		, m_size_total(0)
	{
	}
	LowLevelSession::~LowLevelSession(){
	}

	void LowLevelSession::on_read_avail(StreamBuffer data){
		PROFILE_ME;

		try {
			m_size_total += data.size();
			if(m_size_total > m_max_request_length){
				DEBUG_THROW(Exception, ST_MESSAGE_TOO_LARGE, sslit("Message too large"));
			}

			Reader::put_encoded_data(STD_MOVE(data));
		} catch(Exception &e){
			LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO,
				"WebSocket::Exception thrown in WebSocket parser: status_code = ", e.status_code(), ", what = ", e.what());
			const AUTO(parent, get_parent());
			if(parent){
				parent->force_shutdown();
			}
		}
	}

	void LowLevelSession::on_data_message_header(OpCode opcode){
		PROFILE_ME;

		m_opcode = opcode;
		m_payload.clear();
	}
	void LowLevelSession::on_data_message_payload(boost::uint64_t /* whole_offset */, StreamBuffer payload){
		PROFILE_ME;

		m_payload.splice(payload);
	}
	bool LowLevelSession::on_data_message_end(boost::uint64_t /* whole_size */){
		PROFILE_ME;

		AUTO(opcode, m_opcode);
		AUTO(payload, STD_MOVE_IDN(m_payload));

		m_size_total = 0;
		m_opcode = OP_INVALID_OPCODE;
		m_payload.clear();

		return on_low_level_data_message(opcode, STD_MOVE(payload));
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

	bool LowLevelSession::shutdown(StatusCode status_code, StreamBuffer additional) NOEXCEPT {
		PROFILE_ME;

		const AUTO(parent, get_parent());
		if(!parent){
			return false;
		}

		try {
			Writer::put_close_message(status_code, STD_MOVE(additional));
			parent->shutdown_read();
			return parent->shutdown_write();
		} catch(std::exception &e){
			LOG_POSEIDON_WARNING("std::exception thrown: what = ", e.what());
			parent->force_shutdown();
			return false;
		}
	}
}

}
