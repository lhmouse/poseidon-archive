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
		: m_sizeExpecting(2), m_state(S_PAYLOAD_SIZE)
	{
	}
	Reader::~Reader(){
		if(m_state != S_PAYLOAD_SIZE){
			LOG_POSEIDON_DEBUG("Now that this reader is to be destroyed, a premature message has to be discarded.");
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
				boost::uint16_t temp16;
				boost::uint64_t temp64;

			case S_PAYLOAD_SIZE:
				// m_payloadSize = 0;
				m_messageId = 0;
				m_payloadOffset = 0;

				m_queue.get(&temp16, 2);
				m_payloadSize = loadLe(temp16);
				if(m_payloadSize == 0xFFFF){
					m_sizeExpecting = 8;
					m_state = S_EX_PAYLOAD_SIZE;
				} else {
					m_sizeExpecting = 2;
					m_state = S_MESSAGE_ID;
				}
				break;

			case S_EX_PAYLOAD_SIZE:
				m_queue.get(&temp64, 8);
				m_payloadSize = loadLe(temp64);

				m_sizeExpecting = 2;
				m_state = S_MESSAGE_ID;
				break;

			case S_MESSAGE_ID:
				LOG_POSEIDON_DEBUG("Payload size = ", m_payloadSize);

				m_queue.get(&temp16, 2);
				m_messageId = loadLe(temp16);

				if(m_messageId != ControlMessage::ID){
					onDataMessageHeader(m_messageId, m_payloadSize);

					m_sizeExpecting = std::min<boost::uint64_t>(m_payloadSize, 4096);
					m_state = S_DATA_PAYLOAD;
				} else {
					m_sizeExpecting = m_payloadSize;
					m_state = S_CONTROL_PAYLOAD;
				}
				break;

			case S_DATA_PAYLOAD:
				temp64 = std::min<boost::uint64_t>(m_queue.size(), m_payloadSize - m_payloadOffset);
				onDataMessagePayload(m_payloadOffset, m_queue.cut(temp64));
				m_payloadOffset += temp64;

				if(m_payloadOffset < m_payloadSize){
					m_sizeExpecting = std::min<boost::uint64_t>(m_payloadSize - m_payloadOffset, 4096);
					// m_state = S_DATA_PAYLOAD;
				} else {
					hasNextRequest = onDataMessageEnd(m_payloadOffset);

					m_sizeExpecting = 2;
					m_state = S_PAYLOAD_SIZE;
				}
				break;

			case S_CONTROL_PAYLOAD:
				{
					ControlMessage req(m_queue.cut(m_payloadSize));
					hasNextRequest = onControlMessage(req.controlCode, req.vintParam, STD_MOVE(req.stringParam));
					m_payloadOffset = m_payloadSize;

					m_sizeExpecting = 2;
					m_state = S_PAYLOAD_SIZE;
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
