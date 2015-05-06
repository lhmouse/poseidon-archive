// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "low_level_session.hpp"
#include "exception.hpp"
#include "control_message.hpp"
#include "../singletons/main_config.hpp"
#include "../log.hpp"
#include "../exception.hpp"
#include "../profiler.hpp"
#include "../endian.hpp"

namespace Poseidon {

namespace Cbpp {
	LowLevelSession::LowLevelSession(UniqueFile socket)
		: TcpSessionBase(STD_MOVE(socket))
		, m_sizeTotal(0), m_sizeExpecting(2), m_state(S_PAYLOAD_LEN)
	{
	}
	LowLevelSession::~LowLevelSession(){
		if(m_state != S_PAYLOAD_LEN){
			LOG_POSEIDON_WARNING("Now that this session is to be destroyed, a premature request has to be discarded.");
		}
	}

	void LowLevelSession::onReadAvail(const void *data, std::size_t size){
		PROFILE_ME;

		try {
			const AUTO(maxRequestLength, MainConfig::getConfigFile().get<boost::uint64_t>("cbpp_max_request_length", 16384));

			m_received.put(data, size);

			for(;;){
				boost::uint64_t sizeTotal;
				bool gotExpected;
				if(m_received.size() < m_sizeExpecting){
					if(m_sizeExpecting > maxRequestLength){
						LOG_POSEIDON_WARNING("Request too large: sizeExpecting = ", m_sizeExpecting);
						DEBUG_THROW(Exception, ST_REQUEST_TOO_LARGE, SSLIT("Request too large"));
					}
					sizeTotal = m_sizeTotal + m_received.size();
					gotExpected = false;
				} else {
					sizeTotal = m_sizeTotal + m_sizeExpecting;
					gotExpected = true;
				}
				if(sizeTotal > maxRequestLength){
					LOG_POSEIDON_WARNING("Request too large: sizeTotal = ", sizeTotal);
					DEBUG_THROW(Exception, ST_REQUEST_TOO_LARGE, SSLIT("Request too large"));
				}
				if(!gotExpected){
					break;
				}
				m_sizeTotal = sizeTotal;

				switch(m_state){
					boost::uint16_t temp16;
					boost::uint64_t temp64;

				case S_PAYLOAD_LEN:
					m_received.get(&temp16, 2);
					m_payloadLen = loadLe(temp16);
					if(m_payloadLen == 0xFFFF){
						m_sizeExpecting = 8;
						m_state = S_EX_PAYLOAD_LEN;
					} else {
						m_sizeExpecting = 2;
						m_state = S_MESSAGE_ID;
					}
					break;

				case S_EX_PAYLOAD_LEN:
					m_received.get(&temp64, 8);
					m_payloadLen = loadLe(temp64);

					m_sizeExpecting = 2;
					m_state = S_MESSAGE_ID;
					break;

				case S_MESSAGE_ID:
					LOG_POSEIDON_DEBUG("Payload length = ", m_payloadLen);

					m_received.get(&temp16, 2);
					m_messageId = loadLe(temp16);

					m_sizeExpecting = m_payloadLen;
					m_state = S_PAYLOAD;
					break;

				case S_PAYLOAD:
					if(m_messageId != ControlMessage::ID){
						onLowLevelRequest(m_messageId, m_received.cut(m_payloadLen));
					} else {
						ControlMessage req(m_received.cut(m_payloadLen));
						onLowLevelControl(static_cast<ControlCode>(req.messageId),
							static_cast<StatusCode>(req.statusCode), STD_MOVE(req.reason));
					}

					m_messageId = 0;
					m_payloadLen = 0;

					m_sizeTotal = 0;
					m_sizeExpecting = 2;
					m_state = S_PAYLOAD_LEN;
					break;

				default:
					LOG_POSEIDON_FATAL("Invalid state: ", static_cast<unsigned>(m_state));
					std::abort();
				}
			}
		} catch(Exception &e){
			LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO,
				"Cbpp::Exception thrown while parsing data, message id = ", m_messageId,
				", statusCode = ", static_cast<int>(e.statusCode()), ", what = ", e.what());
			try {
				onLowLevelError(m_messageId, e.statusCode(), e.what());
				shutdownRead();
				shutdownWrite();
			} catch(...){
				forceShutdown();
			}
		} catch(std::exception &e){
			LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO,
				"std::exception thrown while parsing data, message id = ", m_messageId, ", what = ", e.what());
			try {
				onLowLevelError(m_messageId, ST_INTERNAL_ERROR, "");
				shutdownRead();
				shutdownWrite();
			} catch(...){
				forceShutdown();
			}
		}
	}

	bool LowLevelSession::send(boost::uint16_t messageId, StreamBuffer payload){
		PROFILE_ME;

		LOG_POSEIDON_DEBUG("Sending frame: messageId = ", messageId, ", size = ", payload.size());
		StreamBuffer frame;
		boost::uint16_t temp16;
		boost::uint64_t temp64;
		if(payload.size() < 0xFFFF){
			storeLe(temp16, payload.size());
			frame.put(&temp16, 2);
		} else {
			storeLe(temp16, 0xFFFF);
			frame.put(&temp16, 2);
			storeLe(temp64, payload.size());
			frame.put(&temp64, 8);
		}
		storeLe(temp16, messageId);
		frame.put(&temp16, 2);
		frame.splice(payload);
		return TcpSessionBase::send(STD_MOVE(frame));
	}

	bool LowLevelSession::sendControl(boost::uint16_t messageId, StatusCode statusCode, std::string reason){
		return send(ControlMessage(messageId, static_cast<int>(statusCode), STD_MOVE(reason)));
	}
}

}
