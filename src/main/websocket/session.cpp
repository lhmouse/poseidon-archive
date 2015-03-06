// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "session.hpp"
#include "exception.hpp"
#include "../http/session.hpp"
#include "../optional_map.hpp"
#include "../singletons/job_dispatcher.hpp"
#include "../singletons/websocket_servlet_depository.hpp"
#include "../log.hpp"
#include "../utilities.hpp"
#include "../endian.hpp"
#include "../job_base.hpp"
#include "../profiler.hpp"

namespace Poseidon {

namespace WebSocket {
	namespace {
		class RequestJob : public JobBase {
		private:
			const boost::weak_ptr<Session> m_session;
			const std::string m_uri;
			const OpCode m_opcode;
			const StreamBuffer m_payload;

		public:
			RequestJob(boost::weak_ptr<Session> session,
				std::string uri, OpCode opcode, StreamBuffer payload)
				: m_session(STD_MOVE(session)), m_uri(STD_MOVE(uri)), m_opcode(opcode), m_payload(STD_MOVE(payload))
			{
			}

		protected:
			boost::weak_ptr<const void> getCategory() const OVERRIDE {
				return m_session;
			}
			void perform() const OVERRIDE {
				PROFILE_ME;

				const boost::shared_ptr<Session> session(m_session);
				try {
					const AUTO(category, session->getCategory());
					const AUTO(servlet, WebSocketServletDepository::get(category, m_uri.c_str()));
					if(!servlet){
						LOG_POSEIDON_WARNING("No servlet in category ", category, " matches URI ", m_uri);
						DEBUG_THROW(Exception, ST_INACCEPTABLE, SharedNts::observe("Unknown URI"));
						return;
					}

					LOG_POSEIDON_DEBUG("Dispatching packet: URI = ", m_uri, ", payload size = ", m_payload.size());
					(*servlet)(session, m_opcode, m_payload);
					session->setTimeout(WebSocketServletDepository::getKeepAliveTimeout());
				} catch(TryAgainLater &){
					throw;
				} catch(Exception &e){
					LOG_POSEIDON_ERROR("WebSocket::Exception thrown in servlet, statusCode = ", e.statusCode(), ", what = ", e.what());
					session->shutdown(e.statusCode(), StreamBuffer(e.what()));
					throw;
				} catch(...){
					LOG_POSEIDON_ERROR("Forwarding exception...");
					session->shutdown(ST_INTERNAL_ERROR);
					throw;
				}
			}
		};
	}

	Session::Session(const boost::shared_ptr<Http::Session> &parent)
		: Http::UpgradedSessionBase(parent)
		, m_state(S_OPCODE)
	{
	}

	void Session::onInitContents(const void *data, std::size_t size){
		(void)data;
		(void)size;
	}
	void Session::onReadAvail(const void *data, std::size_t size){
		PROFILE_ME;

		m_payload.put(data, size);
		try {
			for(;;){
				switch(m_state){
					int ch;
					boost::uint16_t temp16;
					boost::uint32_t temp32;
					boost::uint64_t temp64;
					std::size_t remaining;

				case S_OPCODE:
					ch = m_payload.get();
					if(ch == -1){
						goto _exitFor;
					}
					if(ch & (OP_FL_RSV1 | OP_FL_RSV2 | OP_FL_RSV3)){
						LOG_POSEIDON_WARNING("Aborting because some reserved bits are set, opcode = ", ch);
						DEBUG_THROW(Exception, ST_PROTOCOL_ERROR, SharedNts::observe("Reserved bits set"));
					}
					m_fin = ch & OP_FL_FIN;
					m_opcode = static_cast<OpCode>(ch & OP_FL_OPCODE);
					if((m_opcode & OP_FL_CONTROL) && !m_fin){
						DEBUG_THROW(Exception, ST_PROTOCOL_ERROR, SharedNts::observe("Control frame fragemented"));
					}
					m_state = S_PAYLOAD_LEN;
					break;

				case S_PAYLOAD_LEN:
					ch = m_payload.get();
					if(ch == -1){
						goto _exitFor;
					}
					if((ch & 0x80) == 0){
						DEBUG_THROW(Exception, ST_ACCESS_DENIED, SharedNts::observe("Non-masked frames not allowed"));
					}
					m_payloadLen = (unsigned char)(ch & 0x7F);
					if(m_payloadLen >= 0x7E){
						if(m_opcode & OP_FL_CONTROL){
							DEBUG_THROW(Exception, ST_PROTOCOL_ERROR, SharedNts::observe("Control frame too large"));
						}
						m_state = S_EX_PAYLOAD_LEN;
					} else {
						m_state = S_MASK;
					}
					break;

				case S_EX_PAYLOAD_LEN:
					if(m_payloadLen == 0x7E){
						if(m_payload.size() < 2){
							goto _exitFor;
						}
						m_payload.get(&temp16, 2);
						m_payloadLen = loadBe(temp16);
					} else {
						if(m_payload.size() < 8){
							goto _exitFor;
						}
						m_payload.get(&temp64, 8);
						m_payloadLen = loadBe(temp64);
					}
					m_state = S_MASK;
					break;

				case S_MASK:
					LOG_POSEIDON_DEBUG("Payload length = ", m_payloadLen);

					if(m_payload.size() < 4){
						goto _exitFor;
					}
					m_payload.get(&temp32, 4);
					m_payloadMask = loadLe(temp32);
					m_state = S_PAYLOAD;
					break;

				case S_PAYLOAD:
					remaining = m_payloadLen - m_whole.size();
					if(m_whole.size() + remaining >= WebSocketServletDepository::getMaxRequestLength()){
						DEBUG_THROW(Exception, ST_MESSAGE_TOO_LARGE, SharedNts::observe("Packet too large"));
					}
					if(m_payload.size() < remaining){
						goto _exitFor;
					}
					for(std::size_t i = 0; i < remaining; ++i){
						ch = m_payload.get();
						ch ^= static_cast<unsigned char>(m_payloadMask);
						m_payloadMask = (m_payloadMask << 24) | (m_payloadMask >> 8);
						m_whole.put(static_cast<unsigned char>(ch));
					}
					if((m_opcode & OP_FL_CONTROL) != 0){
						onControlFrame();
					} else if(m_fin){
						enqueueJob(boost::make_shared<RequestJob>(
							virtualWeakFromThis<Session>(), getUri(), m_opcode, STD_MOVE(m_whole)));
						m_whole.clear();
					}
					m_state = S_OPCODE;
					break;
				}
			}
		_exitFor:
			;
		} catch(Exception &e){
			LOG_POSEIDON_ERROR("Websocket::Exception thrown while parseing data, statusCode = ", e.statusCode(), ", what = ", e.what());
			shutdown(e.statusCode(), StreamBuffer(e.what()));
			throw;
		} catch(...){
			LOG_POSEIDON_ERROR("Forwarding exception... shutdown the session first.");
			shutdown(ST_INTERNAL_ERROR);
			throw;
		}
	}

	void Session::onControlFrame(){
		LOG_POSEIDON_DEBUG("Control frame, opcode = ", m_opcode);

		const AUTO(parent, getSafeParent());

		switch(m_opcode){
		case OP_CLOSE:
			LOG_POSEIDON_INFO("Received close frame from ", parent->getRemoteInfo());
			sendFrame(STD_MOVE(m_whole), OP_CLOSE, true, false);
			break;

		case OP_PING:
			LOG_POSEIDON_INFO("Received ping frame from ", parent->getRemoteInfo());
			sendFrame(STD_MOVE(m_whole), OP_PONG, false, false);
			break;

		case OP_PONG:
			LOG_POSEIDON_INFO("Received pong frame from ", parent->getRemoteInfo());
			break;

		default:
			DEBUG_THROW(Exception, ST_PROTOCOL_ERROR, SharedNts::observe("Invalid opcode"));
			break;
		}
	}

	bool Session::sendFrame(StreamBuffer contents, OpCode opcode, bool fin, bool masked){
		StreamBuffer packet;
		unsigned char ch = opcode | OP_FL_FIN;
		packet.put(ch);
		const std::size_t size = contents.size();
		ch = masked ? 0x80 : 0;
		if(size < 0x7E){
			ch |= size;
			packet.put(ch);
		} else if(size < 0x10000){
			ch |= 0x7E;
			packet.put(ch);
			boost::uint16_t temp;
			storeBe(temp, size);
			packet.put(&temp, 2);
		} else {
			ch |= 0x7F;
			packet.put(ch);
			boost::uint64_t temp;
			storeBe(temp, size);
			packet.put(&temp, 8);
		}
		if(masked){
			boost::uint32_t mask;
			storeLe(mask, rand32());
			packet.put(&mask, 4);
			int ch;
			for(;;){
				ch = contents.get();
				if(ch == -1){
					break;
				}
				ch ^= static_cast<unsigned char>(mask);
				packet.put(ch);
				mask = (mask << 24) | (mask >> 8);
			}
		} else {
			packet.splice(contents);
		}
		return Http::UpgradedSessionBase::send(STD_MOVE(packet), fin);
	}

	bool Session::send(StreamBuffer contents, bool binary, bool fin, bool masked){
		if(!sendFrame(STD_MOVE(contents), binary ? OP_DATA_BIN : OP_DATA_TEXT, false, masked)){
			return false;
		}
		if(fin && !shutdown(ST_NORMAL_CLOSURE)){
			return false;
		}
		return true;
	}
	bool Session::shutdown(StatusCode statusCode, StreamBuffer additional){
		StreamBuffer temp;
		boost::uint16_t codeBe;
		storeBe(codeBe, static_cast<unsigned>(statusCode));
		temp.put(&codeBe, 2);
		temp.splice(additional);
		return sendFrame(STD_MOVE(temp), OP_CLOSE, true, false);
	}
}

}
