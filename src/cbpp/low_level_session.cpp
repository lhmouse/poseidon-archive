// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2016, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "low_level_session.hpp"
#include "exception.hpp"
#include "control_message.hpp"
#include "../singletons/main_config.hpp"
#include "../log.hpp"
#include "../profiler.hpp"
#include "../time.hpp"

namespace Poseidon {

namespace Cbpp {
	LowLevelSession::LowLevelSession(UniqueFile socket, boost::uint64_t max_request_length)
		: TcpSessionBase(STD_MOVE(socket))
		, m_max_request_length(max_request_length ? max_request_length
		                                          : MainConfig::get<boost::uint64_t>("cbpp_max_request_length", 16384))
		, m_size_total(0), m_message_id(0), m_payload()
	{
	}
	LowLevelSession::~LowLevelSession(){
	}

	void LowLevelSession::on_read_avail(StreamBuffer data){
		PROFILE_ME;

		try {
			m_size_total += data.size();
			if(m_size_total > m_max_request_length){
				DEBUG_THROW(Exception, ST_REQUEST_TOO_LARGE);
			}

			Reader::put_encoded_data(STD_MOVE(data));
		} catch(Exception &e){
			LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO,
				"Cbpp::Exception thrown: status_code = ", e.status_code(), ", what = ", e.what());
			force_shutdown();
		}
	}

	void LowLevelSession::on_data_message_header(boost::uint16_t message_id, boost::uint64_t /* payload_size */){
		PROFILE_ME;

		m_message_id = message_id;
		m_payload.clear();
	}
	void LowLevelSession::on_data_message_payload(boost::uint64_t /* payload_offset */, StreamBuffer payload){
		PROFILE_ME;

		m_payload.splice(payload);
	}
	bool LowLevelSession::on_data_message_end(boost::uint64_t /* payload_size */){
		PROFILE_ME;

		AUTO(message_id, m_message_id);
		AUTO(payload, STD_MOVE_IDN(m_payload));

		m_size_total = 0;
		m_message_id = 0;
		m_payload.clear();

		return on_low_level_data_message(message_id, STD_MOVE(payload));
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
