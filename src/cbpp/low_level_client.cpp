// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "low_level_client.hpp"
#include "exception.hpp"
#include "control_message.hpp"
#include "../singletons/timer_daemon.hpp"
#include "../log.hpp"
#include "../exception.hpp"
#include "../profiler.hpp"
#include "../endian.hpp"

namespace Poseidon {

namespace Cbpp {
	namespace {
		void keepAliveTimer(const boost::weak_ptr<LowLevelClient> &weakClient){
			PROFILE_ME;

			const AUTO(client, weakClient.lock());
			if(!client){
				return;
			}
			client->send(ControlMessage(ControlMessage::ID, 0, VAL_INIT));
		}
	}

	LowLevelClient::LowLevelClient(const IpPort &addr, boost::uint64_t keepAliveTimeout, bool useSsl)
		: TcpClientBase(addr, useSsl)
		, m_keepAliveTimeout(keepAliveTimeout)
		, m_sizeExpecting(2), m_state(S_PAYLOAD_LEN)
	{
	}
	LowLevelClient::~LowLevelClient(){
		if(m_state != S_PAYLOAD_LEN){
			LOG_POSEIDON_WARNING("Now that this client is to be destroyed, a premature response has to be discarded.");
		}
	}

	void LowLevelClient::onReadAvail(const void *data, std::size_t size){
		PROFILE_ME;

		try {
			m_received.put(data, size);

			for(;;){
				if(m_received.size() < m_sizeExpecting){
					break;
				}

				switch(m_state){
					boost::uint16_t temp16;
					boost::uint64_t temp64;

				case S_PAYLOAD_LEN:
					// m_payloadLen = 0;
					m_messageId = 0;
					m_payloadOffset = 0;

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

					onLowLevelResponse(m_messageId, m_payloadLen);

					if(m_messageId != ControlMessage::ID){
						m_sizeExpecting = std::min<boost::uint64_t>(m_payloadLen, 1024);
					} else {
						m_sizeExpecting = m_payloadLen;
					}
					m_state = S_PAYLOAD;
					break;

				case S_PAYLOAD:
					if(m_messageId != ControlMessage::ID){
						const AUTO(bytesAvail, std::min<boost::uint64_t>(m_received.size(), m_payloadLen - m_payloadOffset));
						onLowLevelPayload(m_payloadOffset, m_received.cut(bytesAvail));
						m_payloadOffset += bytesAvail;
					} else {
						ControlMessage req(m_received.cut(m_payloadLen));
						onLowLevelError(req.messageId, req.statusCode, STD_MOVE(req.reason));
						m_payloadOffset = m_payloadLen;
					}

					if(m_payloadOffset < m_payloadLen){
						m_sizeExpecting = std::min<boost::uint64_t>(m_payloadLen - m_payloadOffset, 1024);
						// m_state = S_PAYLOAD;
					} else {
						m_sizeExpecting = 2;
						m_state = S_PAYLOAD_LEN;
					}
					break;

				default:
					LOG_POSEIDON_FATAL("Invalid state: ", static_cast<unsigned>(m_state));
					std::abort();
				}
			}
		} catch(std::exception &e){
			LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO,
				"std::exception thrown while parsing data, messageId = ", m_messageId, ", what = ", e.what());
			forceShutdown();
		}
	}

	bool LowLevelClient::send(boost::uint16_t messageId, StreamBuffer payload){
		PROFILE_ME;
		LOG_POSEIDON_DEBUG("Sending frame: messageId = ", messageId, ", size = ", payload.size());

		if(!m_keepAliveTimer){
			m_keepAliveTimer = TimerDaemon::registerTimer(m_keepAliveTimeout, m_keepAliveTimeout,
				boost::bind(&keepAliveTimer, virtualWeakFromThis<LowLevelClient>()), true);
		}

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

	bool LowLevelClient::sendControl(ControlCode controlCode, boost::int64_t intParam, std::string strParam){
		PROFILE_ME;
		LOG_POSEIDON_DEBUG("Sending control frame: controlCode = ", controlCode, ", intParam = ", intParam, ", strParam = ", strParam);

		return send(ControlMessage(controlCode, intParam, STD_MOVE(strParam)));
	}
}

}
