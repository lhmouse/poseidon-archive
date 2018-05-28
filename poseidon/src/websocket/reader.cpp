// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "reader.hpp"
#include "exception.hpp"
#include "../log.hpp"
#include "../random.hpp"
#include "../endian.hpp"
#include "../profiler.hpp"
#include "../flags.hpp"

namespace Poseidon {
namespace Websocket {

Reader::Reader(bool force_masked_frames)
	: m_force_masked_frames(force_masked_frames)
	, m_size_expecting(1), m_state(state_opcode)
	, m_whole_offset(0), m_prev_fin(true)
{
	//
}
Reader::~Reader(){
	if(m_state != state_opcode){
		POSEIDON_LOG_DEBUG("Now that this reader is to be destroyed, a premature request has to be discarded.");
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
			int ch;
			std::uint16_t temp16;
			std::uint32_t temp32;
			std::uint64_t temp64;

		case state_opcode:
			m_fin = false;
			m_masked = false;
			m_opcode = opcode_invalid;
			m_frame_size = 0;
			m_mask = 0;
			m_frame_offset = 0;

			ch = m_queue.get();
			POSEIDON_THROW_UNLESS(has_none_flags_of(ch, opmask_rsv1 | opmask_rsv2 | opmask_rsv3), Exception, status_protocol_error, Rcnts::view("Reserved bits set"));
			m_opcode = ch & opmask_opcode;
			m_fin = ch & opmask_fin;
			POSEIDON_THROW_UNLESS(!(has_all_flags_of(m_opcode, opmask_control) && !m_fin), Exception, status_protocol_error, Rcnts::view("Control frame fragemented"));
			POSEIDON_THROW_UNLESS(!((m_opcode == opcode_continuation) && m_prev_fin), Exception, status_protocol_error, Rcnts::view("Dangling frame continuation"));
			POSEIDON_THROW_UNLESS(!((m_opcode != opcode_continuation) && !m_prev_fin), Exception, status_protocol_error, Rcnts::view("Final frame following a frame that needs continuation"));

			m_size_expecting = 1;
			m_state = state_frame_size;
			break;

		case state_frame_size:
			ch = m_queue.get();
			m_masked = ch & 0x80;
			POSEIDON_THROW_UNLESS(!(m_force_masked_frames && !m_masked), Exception, status_protocol_error, Rcnts::view("Non-masked frames not allowed"));
			m_frame_size = ch & 0x7F;
			if(m_frame_size >= 0x7E){
				POSEIDON_THROW_UNLESS(has_none_flags_of(m_opcode, opmask_control), Exception, status_protocol_error, Rcnts::view("Control frame too large"));
				if(m_frame_size == 0x7E){
					m_size_expecting = 2;
					m_state = state_frame_size_16;
				} else {
					m_size_expecting = 8;
					m_state = state_frame_size_64;
				}
			} else {
				m_size_expecting = 0;
				m_state = state_size_end;
			}
			break;

		case state_frame_size_16:
			m_queue.get(&temp16, 2);
			m_frame_size = load_be(temp16);

			m_size_expecting = 0;
			m_state = state_size_end;
			break;

		case state_frame_size_64:
			m_queue.get(&temp64, 8);
			m_frame_size = load_be(temp64);

			m_size_expecting = 0;
			m_state = state_size_end;
			break;

		case state_size_end:
			POSEIDON_LOG_DEBUG("Frame size = ", m_frame_size);

			if(m_masked){
				m_size_expecting = 4;
				m_state = state_mask;
			} else {
				m_size_expecting = 0;
				m_state = state_header_end;
			}
			break;

		case state_mask:
			m_queue.get(&temp32, 4);
			m_mask = load_le(temp32);

			m_size_expecting = 0;
			m_state = state_header_end;
			break;

		case state_header_end:
			if(m_opcode != opcode_continuation){
				on_data_message_header(m_opcode);
			}

			if(has_none_flags_of(m_opcode, opmask_control)){
				m_size_expecting = std::min<std::uint64_t>(m_frame_size, 4096);
				m_state = state_data_frame;
			} else {
				m_size_expecting = m_frame_size;
				m_state = state_control_frame;
			}
			break;

		case state_data_frame:
			temp64 = std::min<std::uint64_t>(m_queue.size(), m_frame_size - m_frame_offset);
			if(temp64 > 0){
				Stream_buffer payload;
				for(std::size_t i = 0; i < temp64; ++i){
					payload.put(m_queue.get() ^ (int)m_mask);
					m_mask = (m_mask << 24) | (m_mask >> 8);
				}
				on_data_message_payload(m_whole_offset, STD_MOVE(payload));
			}
			m_frame_offset += temp64;
			m_whole_offset += temp64;

			if(m_frame_offset < m_frame_size){
				m_size_expecting = std::min<std::uint64_t>(m_frame_size - m_frame_offset, 4096);
				// m_state = state_data_frame;
			} else {
				if(m_fin){
					has_next_request = on_data_message_end(m_whole_offset);
					m_whole_offset = 0;
					m_prev_fin = true;
				} else {
					m_prev_fin = false;
				}

				m_size_expecting = 1;
				m_state = state_opcode;
			}
			break;

		case state_control_frame:
			{
				Stream_buffer payload;
				for(std::size_t i = 0; i < m_frame_size; ++i){
					payload.put(m_queue.get() ^ (int)m_mask);
					m_mask = (m_mask << 24) | (m_mask >> 8);
				}
				has_next_request = on_control_message(m_opcode, STD_MOVE(payload));
			}
			m_frame_offset = m_frame_size;
			m_whole_offset = 0;
			m_prev_fin = true;

			m_size_expecting = 1;
			m_state = state_opcode;
			break;
		}
	} while(has_next_request);

	return has_next_request;
}

}
}
