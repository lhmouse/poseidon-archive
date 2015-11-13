// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "reader.hpp"
#include "control_message.hpp"
#include "../log.hpp"
#include "../profiler.hpp"
#include "../endian.hpp"

namespace Poseidon {

namespace Cbpp {
	Reader::Reader()
		: m_size_expecting(2), m_state(S_PAYLOAD_SIZE)
	{
	}
	Reader::~Reader(){
		if(m_state != S_PAYLOAD_SIZE){
			LOG_POSEIDON_DEBUG("Now that this reader is to be destroyed, a premature message has to be discarded.");
		}
	}

	bool Reader::put_encoded_data(StreamBuffer encoded){
		PROFILE_ME;

		m_queue.splice(encoded);

		bool has_next_request = true;
		do {
			if(m_queue.size() < m_size_expecting){
				break;
			}

			switch(m_state){
				boost::uint16_t temp16;
				boost::uint64_t temp64;

			case S_PAYLOAD_SIZE:
				// m_payload_size = 0;
				m_message_id = 0;
				m_payload_offset = 0;

				m_queue.get(&temp16, 2);
				m_payload_size = load_le(temp16);
				if(m_payload_size == 0xFFFF){
					m_size_expecting = 8;
					m_state = S_EX_PAYLOAD_SIZE;
				} else {
					m_size_expecting = 2;
					m_state = S_MESSAGE_ID;
				}
				break;

			case S_EX_PAYLOAD_SIZE:
				m_queue.get(&temp64, 8);
				m_payload_size = load_le(temp64);

				m_size_expecting = 2;
				m_state = S_MESSAGE_ID;
				break;

			case S_MESSAGE_ID:
				LOG_POSEIDON_DEBUG("Payload size = ", m_payload_size);

				m_queue.get(&temp16, 2);
				m_message_id = load_le(temp16);

				if(m_message_id != ControlMessage::ID){
					on_data_message_header(m_message_id, m_payload_size);

					m_size_expecting = std::min<boost::uint64_t>(m_payload_size, 4096);
					m_state = S_DATA_PAYLOAD;
				} else {
					m_size_expecting = m_payload_size;
					m_state = S_CONTROL_PAYLOAD;
				}
				break;

			case S_DATA_PAYLOAD:
				temp64 = std::min<boost::uint64_t>(m_queue.size(), m_payload_size - m_payload_offset);
				on_data_message_payload(m_payload_offset, m_queue.cut_off(temp64));
				m_payload_offset += temp64;

				if(m_payload_offset < m_payload_size){
					m_size_expecting = std::min<boost::uint64_t>(m_payload_size - m_payload_offset, 4096);
					// m_state = S_DATA_PAYLOAD;
				} else {
					has_next_request = on_data_message_end(m_payload_offset);

					m_size_expecting = 2;
					m_state = S_PAYLOAD_SIZE;
				}
				break;

			case S_CONTROL_PAYLOAD:
				{
					ControlMessage req(m_queue.cut_off(m_payload_size));
					has_next_request = on_control_message(req.control_code, req.vint_param, STD_MOVE(req.string_param));
					m_payload_offset = m_payload_size;

					m_size_expecting = 2;
					m_state = S_PAYLOAD_SIZE;
				}
				break;

			default:
				LOG_POSEIDON_FATAL("Invalid state: ", static_cast<unsigned>(m_state));
				std::abort();
			}
		} while(has_next_request);

		return has_next_request;
	}
}

}
