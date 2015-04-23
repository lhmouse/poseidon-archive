// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "session.hpp"
#include <iomanip>
#include "exception.hpp"
#include "../http/session.hpp"
#include "../http/utilities.hpp"
#include "../optional_map.hpp"
#include "../singletons/main_config.hpp"
#include "../log.hpp"
#include "../random.hpp"
#include "../endian.hpp"
#include "../job_base.hpp"
#include "../profiler.hpp"
#include "../hash.hpp"

namespace Poseidon {

namespace WebSocket {
	namespace {
		class SessionJobBase : public JobBase {
		private:
			const boost::weak_ptr<Http::Session> m_parent;
			const boost::weak_ptr<Session> m_session;

		protected:
			explicit SessionJobBase(const boost::shared_ptr<Session> &session)
				: m_parent(session->getWeakParent()), m_session(session)
			{
			}

		protected:
			virtual void perform(const boost::shared_ptr<Session> &session) const = 0;

		private:
			boost::weak_ptr<const void> getCategory() const FINAL {
				return m_parent;
			}
			void perform() const FINAL {
				PROFILE_ME;

				const AUTO(session, m_session.lock());
				if(!session){
					return;
				}

				try {
					perform(session);
				} catch(TryAgainLater &){
					throw;
				} catch(std::exception &e){
					LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "std::exception thrown: what = ", e.what());
					session->forceShutdown();
					throw;
				} catch(...){
					LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "Unknown exception thrown.");
					session->forceShutdown();
					throw;
				}
			}
		};

		bool isValidUtf8String(const std::string &str){
			PROFILE_ME;

			boost::uint32_t codePoint;
			for(AUTO(it, str.begin()); it != str.end(); ++it){
				codePoint = static_cast<unsigned char>(*it);
				if((codePoint & 0x80) == 0){
					continue;
				}
				const AUTO(bytes, static_cast<unsigned>(
					__builtin_clz((~codePoint | 1) & 0xFF)) - (sizeof(unsigned) - sizeof(unsigned char)) * CHAR_BIT);
				if(bytes - 2 > 2){ // 2, 3, 4
					LOG_POSEIDON_WARNING("Invalid UTF-8 leading byte: bytes = ", bytes);
					return false;
				}
				codePoint &= (0xFFu >> bytes);
				for(unsigned i = 1; i < bytes; ++i){
					++it;
					if(it == str.end()){
						LOG_POSEIDON_WARNING("String is truncated.");
						return false;
					}
					const unsigned trailing = static_cast<unsigned char>(*it);
					if((trailing & 0xC0u) != 0x80u){
						LOG_POSEIDON_WARNING("Invalid UTF-8 trailing byte: trailing = 0x",
							std::hex, std::setw(2), std::setfill('0'), trailing);
						return false;
					}
				}
				if(codePoint > 0x10FFFFu){
					LOG_POSEIDON_WARNING("Invalid UTF-8 code point: codePoint = 0x",
						std::hex, std::setw(6), std::setfill('0'), codePoint);
					return false;
				}
				if(codePoint - 0xD800u < 0x800u){
					LOG_POSEIDON_WARNING("UTF-8 code point is reserved for UTF-16: codePoint = 0x",
						std::hex, std::setw(4), std::setfill('0'), codePoint);
					return false;
				}
			}
			return true;
		}
	}

	class Session::RequestJob : public SessionJobBase {
	private:
		const OpCode m_opcode;
		const StreamBuffer m_payload;

	public:
		RequestJob(const boost::shared_ptr<Session> &session, OpCode opcode, StreamBuffer payload)
			: SessionJobBase(session)
			, m_opcode(opcode), m_payload(STD_MOVE(payload))
		{
		}

	protected:
		void perform(const boost::shared_ptr<Session> &session) const OVERRIDE {
			PROFILE_ME;

			try {
				LOG_POSEIDON_DEBUG("Dispatching request: URI = ", session->getUri(), ", payload size = ", m_payload.size());
				session->onRequest(m_opcode, m_payload);
				session->setTimeout(MainConfig::getConfigFile().get<boost::uint64_t>("websocket_keep_alive_timeout", 0));
			} catch(TryAgainLater &){
				throw;
			} catch(Exception &e){
				LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "WebSocket::Exception thrown: URI = ", session->getUri(),
					", statusCode = ", e.statusCode(), ", what = ", e.what());
				try {
					session->shutdown(e.statusCode(), StreamBuffer(e.what()));
				} catch(...){
					session->forceShutdown();
				}
			}
		}
	};

	class Session::ErrorJob : public SessionJobBase {
	private:
		const StatusCode m_statusCode;
		const StreamBuffer m_additional;

	public:
		ErrorJob(const boost::shared_ptr<Session> &session, StatusCode statusCode, StreamBuffer additional)
			: SessionJobBase(session)
			, m_statusCode(statusCode), m_additional(STD_MOVE(additional))
		{
		}

	protected:
		void perform(const boost::shared_ptr<Session> &session) const OVERRIDE {
			PROFILE_ME;

			session->shutdown(m_statusCode, m_additional);
		}
	};

	Http::StatusCode Session::makeHttpHandshakeResponse(OptionalMap &ret,
		Http::Verb verb, unsigned version, const OptionalMap &headers)
	{
		if(verb != Http::V_GET){
			LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "Must use GET verb to use WebSocket");
			return Http::ST_METHOD_NOT_ALLOWED;
		}
		if(version < 10001){
			LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "HTTP 1.1 is required to use WebSocket");
			return Http::ST_VERSION_NOT_SUPPORTED;
		}
		AUTO_REF(wsVersion, headers.get("Sec-WebSocket-Version"));
		if(wsVersion != "13"){
			LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "Unknown HTTP header Sec-WebSocket-Version: ", wsVersion);
			return Http::ST_BAD_REQUEST;
		}

		std::string key = headers.get("Sec-WebSocket-Key");
		if(key.empty()){
			LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "No Sec-WebSocket-Key specified.");
			return Http::ST_BAD_REQUEST;
		}
		key += "258EAFA5-E914-47DA-95CA-C5AB0DC85B11";
		unsigned char sha1[20];
		sha1Sum(sha1, key.data(), key.size());
		key = Http::base64Encode(sha1, sizeof(sha1));

		ret.set("Upgrade", "websocket");
		ret.set("Connection", "Upgrade");
		ret.set("Sec-WebSocket-Accept", STD_MOVE(key));
		return Http::ST_OK;
	}

	Session::Session(const boost::shared_ptr<Http::Session> &parent, std::string uri)
		: Http::UpgradedSessionBase(parent, STD_MOVE(uri))
		, m_sizeTotal(0), m_sizeExpecting(1), m_state(S_OPCODE)
		, m_fin(false), m_opcode(OP_INVALID_OPCODE), m_payloadLen(0), m_payloadMask(0)
	{
	}
	Session::~Session(){
		if(m_state != S_OPCODE){
			LOG_POSEIDON_WARNING("Now that this session is to be destroyed, a premature request has to be discarded.");
		}
	}

	void Session::onReadAvail(const void *data, std::size_t size){
		PROFILE_ME;

		try {
			const AUTO(maxRequestLength, MainConfig::getConfigFile().get<boost::uint64_t>("websocket_max_request_length", 16384));

			m_received.put(data, size);

			for(;;){
				boost::uint64_t sizeTotal;
				bool gotExpected;
				if(m_received.size() < m_sizeExpecting){
					if(m_sizeExpecting > maxRequestLength){
						LOG_POSEIDON_WARNING("Message too large: sizeExpecting = ", m_sizeExpecting);
						DEBUG_THROW(Exception, ST_MESSAGE_TOO_LARGE, SSLIT("Message too large"));
					}
					sizeTotal = m_sizeTotal + m_received.size();
					gotExpected = false;
				} else {
					sizeTotal = m_sizeTotal + m_sizeExpecting;
					gotExpected = true;
				}
				if(sizeTotal > maxRequestLength){
					LOG_POSEIDON_WARNING("Message too large: sizeTotal = ", sizeTotal);
					DEBUG_THROW(Exception, ST_MESSAGE_TOO_LARGE, SSLIT("Message too large"));
				}
				if(!gotExpected){
					break;
				}
				m_sizeTotal = sizeTotal;

				switch(m_state){
					int ch;
					boost::uint16_t temp16;
					boost::uint32_t temp32;
					boost::uint64_t temp64;

				case S_OPCODE:
					ch = m_received.get();
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
					m_state = S_PAYLOAD_LEN;
					break;

				case S_PAYLOAD_LEN:
					ch = m_received.get();
					if((ch & 0x80) == 0){
						DEBUG_THROW(Exception, ST_ACCESS_DENIED, SSLIT("Non-masked frames not allowed"));
					}
					m_payloadLen = static_cast<unsigned char>(ch & 0x7F);
					if(m_payloadLen >= 0x7E){
						if(m_opcode & OP_FL_CONTROL){
							DEBUG_THROW(Exception, ST_PROTOCOL_ERROR, SSLIT("Control frame too large"));
						}
						if(m_payloadLen == 0x7E){
							m_sizeExpecting = 2;
							m_state = S_EX_PAYLOAD_LEN_16;
						} else {
							m_sizeExpecting = 8;
							m_state = S_EX_PAYLOAD_LEN_64;
						}
					} else {
						m_sizeExpecting = 4;
						m_state = S_MASK;
					}
					break;

				case S_EX_PAYLOAD_LEN_16:
					m_received.get(&temp16, 2);
					m_payloadLen = loadBe(temp16);

					m_sizeExpecting = 4;
					m_state = S_MASK;
					break;

				case S_EX_PAYLOAD_LEN_64:
					m_received.get(&temp64, 8);
					m_payloadLen = loadBe(temp64);

					m_sizeExpecting = 4;
					m_state = S_MASK;
					break;

				case S_MASK:
					LOG_POSEIDON_DEBUG("Payload length = ", m_payloadLen);

					m_received.get(&temp32, 4);
					m_payloadMask = loadLe(temp32);

					m_sizeExpecting = m_payloadLen;
					m_state = S_PAYLOAD;
					break;

				case S_PAYLOAD:
					{
						StreamBuffer payload;
						for(boost::uint64_t i = 0; i < m_payloadLen; ++i){
							payload.put(static_cast<unsigned char>(m_received.get()) ^ m_payloadMask);
							m_payloadMask = (m_payloadMask << 24) | (m_payloadMask >> 8);
						}
						if((m_opcode & OP_FL_CONTROL) != 0){
							LOG_POSEIDON_DEBUG("Control frame, opcode = ", m_opcode);

							const AUTO(parent, getSafeParent());
							switch(m_opcode){
							case OP_CLOSE:
								LOG_POSEIDON_INFO("Received close frame from ", parent->getRemoteInfo());
								sendFrame(STD_MOVE(payload), OP_CLOSE, true, false);
								break;

							case OP_PING:
								LOG_POSEIDON_INFO("Received ping frame from ", parent->getRemoteInfo());
								sendFrame(STD_MOVE(payload), OP_PONG, false, false);
								break;

							case OP_PONG:
								LOG_POSEIDON_INFO("Received pong frame from ", parent->getRemoteInfo());
								break;

							default:
								DEBUG_THROW(Exception, ST_PROTOCOL_ERROR, SSLIT("Invalid opcode"));
								break;
							}
						} else if(m_fin){
							if(m_opcode == OP_DATA_TEXT){
								LOG_POSEIDON_DEBUG("Validating UTF-8 string...");

								std::string temp;
								payload.dump(temp);
								if(!isValidUtf8String(temp)){
									DEBUG_THROW(Exception, ST_INCONSISTENT, SSLIT("Invalid UTF-8 string"));
								}
							}

							enqueueJob(boost::make_shared<RequestJob>(
								virtualSharedFromThis<Session>(), m_opcode, STD_MOVE(payload)));
						}
					}

					m_fin = false;
					m_opcode = OP_INVALID_OPCODE;
					m_payloadLen = 0;

					m_sizeTotal = 0;
					m_sizeExpecting = 1;
					m_state = S_OPCODE;
					break;

				default:
					LOG_POSEIDON_FATAL("Invalid state: ", static_cast<unsigned>(m_state));
					std::abort();
				}
			}
		} catch(Exception &e){
			LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "WebSocket::Exception thrown while parsing data: URI = ", getUri(),
				", statusCode = ", e.statusCode(), ", what = ", e.what());
			const AUTO(parent, getParent());
			if(parent){
				try {
					enqueueJob(boost::make_shared<ErrorJob>(
						virtualSharedFromThis<Session>(), e.statusCode(), StreamBuffer(e.what())));
					parent->setPreservedOnReadHup(true); // noexcept
					parent->shutdown();
				} catch(...){
					parent->forceShutdown();
				}
			}
		} catch(std::exception &e){
			LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "std::exception thrown while parsing data: URI = ", getUri(),
				", what = ", e.what());
			const AUTO(parent, getParent());
			if(parent){
				try {
					enqueueJob(boost::make_shared<ErrorJob>(
						virtualSharedFromThis<Session>(), ST_INTERNAL_ERROR, StreamBuffer()));
					parent->setPreservedOnReadHup(true); // noexcept
					parent->shutdown();
				} catch(...){
					forceShutdown();
				}
			}
		}
	}

	bool Session::sendFrame(StreamBuffer contents, OpCode opcode, bool fin, bool masked){
		StreamBuffer frame;
		unsigned char ch = opcode | OP_FL_FIN;
		frame.put(ch);
		const std::size_t size = contents.size();
		ch = masked ? 0x80 : 0;
		if(size < 0x7E){
			ch |= size;
			frame.put(ch);
		} else if(size < 0x10000){
			ch |= 0x7E;
			frame.put(ch);
			boost::uint16_t temp;
			storeBe(temp, size);
			frame.put(&temp, 2);
		} else {
			ch |= 0x7F;
			frame.put(ch);
			boost::uint64_t temp;
			storeBe(temp, size);
			frame.put(&temp, 8);
		}
		if(masked){
			boost::uint32_t mask;
			storeLe(mask, rand32() | 0x80808080u);
			frame.put(&mask, 4);
			int ch;
			for(;;){
				ch = contents.get();
				if(ch == -1){
					break;
				}
				ch ^= static_cast<unsigned char>(mask);
				frame.put(ch);
				mask = (mask << 24) | (mask >> 8);
			}
		} else {
			frame.splice(contents);
		}
		return Http::UpgradedSessionBase::send(STD_MOVE(frame), fin);
	}

	bool Session::send(StreamBuffer contents, bool binary, bool fin, bool masked){
		if(!sendFrame(STD_MOVE(contents), binary ? OP_DATA_BIN : OP_DATA_TEXT, false, masked)){
			return false;
		}
		if(fin){
			return shutdown(ST_NORMAL_CLOSURE);
		}
		return true;
	}
	bool Session::shutdown(StatusCode statusCode, StreamBuffer additional) NOEXCEPT {
		try {
			StreamBuffer temp;
			boost::uint16_t codeBe;
			storeBe(codeBe, static_cast<unsigned>(statusCode));
			temp.put(&codeBe, 2);
			temp.splice(additional);
			return sendFrame(STD_MOVE(temp), OP_CLOSE, true, false);
		} catch(...){
			return UpgradedSessionBase::shutdown();
		}
	}
}

}
