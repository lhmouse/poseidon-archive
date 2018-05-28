// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "reader.hpp"
#include "status_codes.hpp"
#include "exception.hpp"
#include "../log.hpp"
#include "../profiler.hpp"
#include "../endian.hpp"

namespace Poseidon {
namespace Cbpp {

Reader::Reader()
	: m_size_expecting(2), m_state(state_payload_size)
{
	//
}
Reader::~Reader(){
	if(m_state != state_payload_size){
		POSEIDON_LOG_DEBUG("Now that this reader is to be destroyed, a premature message has to be discarded.");
	}
}

bool Reader::put_encoded_data(Stream_buffer encoded){
	POSEIDON_PROFILE_ME;

	m_queue.splice(encoded);

	bool has_next_request = true;
	do {
		if(m_queue.size() < m_size_expecting){
			break;
		}

		switch(m_state){
			std::uint16_t temp16;
			std::uint64_t temp64;

		case state_payload_size:
			// m_payload_size = 0;
			m_message_id = 0;
			m_payload_offset = 0;

			m_queue.get(&temp16, 2);
			m_payload_size = load_be(temp16);
			if(m_payload_size == 0xFFFF){
				m_size_expecting = 8;
				m_state = state_ex_payload_size;
			} else {
				m_size_expecting = 2;
				m_state = state_message_id;
			}
			break;

		case state_ex_payload_size:
			m_queue.get(&temp64, 8);
			m_payload_size = load_be(temp64);

			m_size_expecting = 2;
			m_state = state_message_id;
			break;

		case state_message_id:
			m_queue.get(&temp16, 2);
			m_message_id = load_be(temp16);

			if(m_message_id != 0){
				on_data_message_header(m_message_id, m_payload_size);

				m_size_expecting = std::min<std::uint64_t>(m_payload_size, 4096);
				m_state = state_data_payload;
			} else {
				m_size_expecting = m_payload_size;
				m_state = state_control_payload;
			}
			break;

		case state_data_payload:
			temp64 = std::min<std::uint64_t>(m_queue.size(), m_payload_size - m_payload_offset);
			if(temp64 > 0){
				on_data_message_payload(m_payload_offset, m_queue.cut_off(boost::numeric_cast<std::size_t>(temp64)));
			}
			m_payload_offset += temp64;

			if(m_payload_offset < m_payload_size){
				m_size_expecting = std::min<std::uint64_t>(m_payload_size - m_payload_offset, 4096);
				// m_state = state_data_payload;
			} else {
				has_next_request = on_data_message_end(m_payload_offset);

				m_size_expecting = 2;
				m_state = state_payload_size;
			}
			break;

		case state_control_payload:
			{
				Stream_buffer payload = m_queue.cut_off(boost::numeric_cast<std::size_t>(m_payload_size));
				std::uint32_t temp32;
				POSEIDON_THROW_UNLESS(payload.get(&temp32, 4) == 4, Exception, status_end_of_stream, Rcnts::view("control.code"));
				has_next_request = on_control_message(static_cast<std::int32_t>(load_be(temp32)), STD_MOVE(payload));
			}
			m_payload_offset = m_payload_size;

			m_size_expecting = 2;
			m_state = state_payload_size;
			break;
		}
	} while(has_next_request);

	return has_next_request;
}

}
}
