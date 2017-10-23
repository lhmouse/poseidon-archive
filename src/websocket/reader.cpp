// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2017, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "reader.hpp"
#include "exception.hpp"
#include "../log.hpp"
#include "../random.hpp"
#include "../endian.hpp"
#include "../profiler.hpp"

namespace Poseidon {
namespace WebSocket {

Reader::Reader(bool force_masked_frames)
	: m_force_masked_frames(force_masked_frames)
	, m_size_expecting(1), m_state(S_OPCODE)
	, m_whole_offset(0), m_prev_fin(true)
{ }
Reader::~Reader(){
	if(m_state != S_OPCODE){
		LOG_POSEIDON_DEBUG("Now that this reader is to be destroyed, a premature request has to be discarded.");
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
			int ch;
			boost::uint16_t temp16;
			boost::uint32_t temp32;
			boost::uint64_t temp64;

		case S_OPCODE:
			m_fin = false;
			m_masked = false;
			m_opcode = OP_INVALID;
			m_frame_size = 0;
			m_mask = 0;
			m_frame_offset = 0;

			ch = m_queue.get();
			if(ch & (OP_FL_RSV1 | OP_FL_RSV2 | OP_FL_RSV3)){
				LOG_POSEIDON_WARNING("Aborting because some reserved bits are set, opcode = ", ch);
				DEBUG_THROW(Exception, ST_PROTOCOL_ERROR, sslit("Reserved bits set"));
			}
			m_opcode = static_cast<OpCode>(ch & OP_FL_OPCODE);
			m_fin = ch & OP_FL_FIN;
			if((m_opcode & OP_FL_CONTROL) && !m_fin){
				DEBUG_THROW(Exception, ST_PROTOCOL_ERROR, sslit("Control frame fragemented"));
			}
			if((m_opcode == OP_CONTINUATION) && m_prev_fin){
				DEBUG_THROW(Exception, ST_PROTOCOL_ERROR, sslit("Dangling frame continuation"));
			}
			if((m_opcode != OP_CONTINUATION) && !m_prev_fin){
				DEBUG_THROW(Exception, ST_PROTOCOL_ERROR, sslit("Final frame following a frame that needs continuation"));
			}

			m_size_expecting = 1;
			m_state = S_FRAME_SIZE;
			break;

		case S_FRAME_SIZE:
			ch = m_queue.get();
			m_masked = ch & 0x80;
			if(m_force_masked_frames && !m_masked){
				DEBUG_THROW(Exception, ST_PROTOCOL_ERROR, sslit("Non-masked frames not allowed"));
			}
			m_frame_size = static_cast<unsigned char>(ch & 0x7F);
			if(m_frame_size >= 0x7E){
				if(m_opcode & OP_FL_CONTROL){
					DEBUG_THROW(Exception, ST_PROTOCOL_ERROR, sslit("Control frame too large"));
				}
				if(m_frame_size == 0x7E){
					m_size_expecting = 2;
					m_state = S_FRAME_SIZE_16;
				} else {
					m_size_expecting = 8;
					m_state = S_FRAME_SIZE_64;
				}
			} else {
				m_size_expecting = 0;
				m_state = S_SIZE_END;
			}
			break;

		case S_FRAME_SIZE_16:
			m_queue.get(&temp16, 2);
			m_frame_size = load_be(temp16);

			m_size_expecting = 0;
			m_state = S_SIZE_END;
			break;

		case S_FRAME_SIZE_64:
			m_queue.get(&temp64, 8);
			m_frame_size = load_be(temp64);

			m_size_expecting = 0;
			m_state = S_SIZE_END;
			break;

		case S_SIZE_END:
			LOG_POSEIDON_DEBUG("Frame size = ", m_frame_size);

			if(m_masked){
				m_size_expecting = 4;
				m_state = S_MASK;
			} else {
				m_size_expecting = 0;
				m_state = S_HEADER_END;
			}
			break;

		case S_MASK:
			m_queue.get(&temp32, 4);
			m_mask = load_le(temp32);

			m_size_expecting = 0;
			m_state = S_HEADER_END;
			break;

		case S_HEADER_END:
			if(m_opcode != OP_CONTINUATION){
				on_data_message_header(m_opcode);
			}

			if((m_opcode & OP_FL_CONTROL) == 0){
				m_size_expecting = std::min<boost::uint64_t>(m_frame_size, 4096);
				m_state = S_DATA_FRAME;
			} else {
				m_size_expecting = m_frame_size;
				m_state = S_CONTROL_FRAME;
			}
			break;

		case S_DATA_FRAME:
			temp64 = std::min<boost::uint64_t>(m_queue.size(), m_frame_size - m_frame_offset);
			{
				StreamBuffer payload;
				for(std::size_t i = 0; i < temp64; ++i){
					payload.put(static_cast<unsigned char>(m_queue.get()) ^ m_mask);
					m_mask = (m_mask << 24) | (m_mask >> 8);
				}
				on_data_message_payload(m_whole_offset, STD_MOVE(payload));
			}
			m_frame_offset += temp64;
			m_whole_offset += temp64;

			if(m_frame_offset < m_frame_size){
				m_size_expecting = std::min<boost::uint64_t>(m_frame_size - m_frame_offset, 4096);
				// m_state = S_DATA_FRAME;
			} else {
				if(m_fin){
					has_next_request = on_data_message_end(m_whole_offset);
					m_whole_offset = 0;
					m_prev_fin = true;
				} else {
					m_prev_fin = false;
				}

				m_size_expecting = 1;
				m_state = S_OPCODE;
			}
			break;

		case S_CONTROL_FRAME:
			{
				StreamBuffer payload;
				for(std::size_t i = 0; i < m_frame_size; ++i){
					payload.put(static_cast<unsigned char>(m_queue.get()) ^ m_mask);
					m_mask = (m_mask << 24) | (m_mask >> 8);
				}
				has_next_request = on_control_message(m_opcode, STD_MOVE(payload));
			}
			m_frame_offset = m_frame_size;
			m_whole_offset = 0;
			m_prev_fin = true;

			m_size_expecting = 1;
			m_state = S_OPCODE;
			break;
		}
	} while(has_next_request);

	return has_next_request;
}

}
}
