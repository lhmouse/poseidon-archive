// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "reader.hpp"
#include "exception.hpp"
#include "../log.hpp"
#include "../random.hpp"
#include "../endian.hpp"
#include "../profiler.hpp"

namespace Poseidon {

namespace WebSocket {
	Reader::Reader()
		: m_sizeExpecting(1), m_state(S_OPCODE)
	{
	}
	Reader::~Reader(){
		if(m_state != S_OPCODE){
			LOG_POSEIDON_WARNING("Now that this reader is to be destroyed, a premature request has to be discarded.");
		}
	}

	bool Reader::putEncodedData(StreamBuffer encoded){
		PROFILE_ME;

		m_queue.splice(encoded);

		bool hasNextRequest = true;
		do {
			if(m_queue.size() < m_sizeExpecting){
				break;
			}

			switch(m_state){
				int ch;
				boost::uint16_t temp16;
				boost::uint32_t temp32;
				boost::uint64_t temp64;

			case S_OPCODE:
				m_wholeOffset = 0;
				m_fin = false;
				m_opcode = OP_INVALID_OPCODE;
				m_frameSize = 0;
				m_mask = 0;
				m_frameOffset = 0;

				ch = m_queue.get();
				if(ch & (OP_FL_RSV1 | OP_FL_RSV2 | OP_FL_RSV3)){
					LOG_POSEIDON_WARNING("Aborting because some reserved bits are set, opcode = ", ch);
					DEBUG_THROW(Exception, ST_PROTOCOL_ERROR, SSLIT("Reserved bits set"));
				}
				m_fin = ch & OP_FL_FIN;
				m_opcode = static_cast<OpCode>(ch & OP_FL_OPCODE);
				if((m_opcode & OP_FL_CONTROL) && !m_fin){
					DEBUG_THROW(Exception, ST_PROTOCOL_ERROR, SSLIT("Control frame fragemented"));
				}

				m_sizeExpecting = 1;
				m_state = S_FRAME_SIZE;
				break;

			case S_FRAME_SIZE:
				ch = m_queue.get();
				if((ch & 0x80) == 0){
					DEBUG_THROW(Exception, ST_ACCESS_DENIED, SSLIT("Non-masked frames not allowed"));
				}
				m_frameSize = static_cast<unsigned char>(ch & 0x7F);
				if(m_frameSize >= 0x7E){
					if(m_opcode & OP_FL_CONTROL){
						DEBUG_THROW(Exception, ST_PROTOCOL_ERROR, SSLIT("Control frame too large"));
					}
					if(m_frameSize == 0x7E){
						m_sizeExpecting = 2;
						m_state = S_FRAME_SIZE_16;
					} else {
						m_sizeExpecting = 8;
						m_state = S_FRAME_SIZE_64;
					}
				} else {
					m_sizeExpecting = 4;
					m_state = S_MASK;
				}
				break;

			case S_FRAME_SIZE_16:
				m_queue.get(&temp16, 2);
				m_frameSize = loadBe(temp16);

				m_sizeExpecting = 4;
				m_state = S_MASK;
				break;

			case S_FRAME_SIZE_64:
				m_queue.get(&temp64, 8);
				m_frameSize = loadBe(temp64);

				m_sizeExpecting = 4;
				m_state = S_MASK;
				break;

			case S_MASK:
				LOG_POSEIDON_DEBUG("Frame size = ", m_frameSize);

				m_queue.get(&temp32, 4);
				m_mask = loadLe(temp32);

				onDataMessageHeader(m_opcode);

				if((m_opcode & OP_FL_CONTROL) == 0){
					m_sizeExpecting = std::min<boost::uint64_t>(m_frameSize, 1024);
					m_state = S_DATA_FRAME;
				} else {
					m_sizeExpecting = m_frameSize;
					m_state = S_CONTROL_FRAME;
				}
				break;

			case S_DATA_FRAME:
				{
					temp64 = std::min<boost::uint64_t>(m_queue.size(), m_frameSize - m_frameOffset);
					StreamBuffer payload;
					for(std::size_t i = 0; i < temp64; ++i){
						payload.put(static_cast<unsigned char>(m_queue.get()) ^ m_mask);
						m_mask = (m_mask << 24) | (m_mask >> 8);
					}
					onDataMessagePayload(m_wholeOffset, STD_MOVE(payload));
					m_wholeOffset += temp64;
					m_frameOffset += temp64;

					if(m_frameOffset < m_frameSize){
						m_sizeExpecting = std::min<boost::uint64_t>(m_frameSize - m_frameOffset, 1024);
						// m_state = S_FRAME;
					} else {
						hasNextRequest = onDataMessageEnd(m_frameOffset);

						m_sizeExpecting = 1;
						m_state = S_OPCODE;
					}
				}
				break;

			case S_CONTROL_FRAME:
				{
					StreamBuffer payload;
					for(std::size_t i = 0; i < m_frameSize; ++i){
						payload.put(static_cast<unsigned char>(m_queue.get()) ^ m_mask);
						m_mask = (m_mask << 24) | (m_mask >> 8);
					}
					hasNextRequest = onControlMessage(m_opcode, STD_MOVE(payload));
					m_wholeOffset = m_frameSize;
					m_frameOffset = m_frameSize;

					m_sizeExpecting = 1;
					m_state = S_OPCODE;
				}
				break;

			default:
				LOG_POSEIDON_FATAL("Invalid state: ", static_cast<unsigned>(m_state));
				std::abort();
			}
		} while(hasNextRequest);

		return hasNextRequest;
	}
}

}
