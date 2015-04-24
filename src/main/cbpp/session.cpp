// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "session.hpp"
#include "exception.hpp"
#include "error_message.hpp"
#include "../singletons/main_config.hpp"
#include "../log.hpp"
#include "../exception.hpp"
#include "../job_base.hpp"
#include "../profiler.hpp"
#include "../endian.hpp"

namespace Poseidon {

namespace Cbpp {
	namespace {
		class SessionJobBase : public JobBase {
		private:
			const boost::weak_ptr<Session> m_session;

		protected:
			explicit SessionJobBase(const boost::shared_ptr<Session> &session)
				: m_session(session)
			{
			}

		protected:
			virtual void perform(const boost::shared_ptr<Session> &session) const = 0;

		private:
			boost::weak_ptr<const void> getCategory() const FINAL {
				return m_session;
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
	}

	class Session::RequestJob : public SessionJobBase {
	private:
		const unsigned m_messageId;
		const StreamBuffer m_payload;

	public:
		RequestJob(const boost::shared_ptr<Session> &session, unsigned messageId, StreamBuffer payload)
			: SessionJobBase(session)
			, m_messageId(messageId), m_payload(STD_MOVE(payload))
		{
		}

	protected:
		void perform(const boost::shared_ptr<Session> &session) const OVERRIDE {
			PROFILE_ME;

			try {
				if(m_messageId == ErrorMessage::ID){
					AUTO(payload, m_payload);
					ErrorMessage req(payload);
					LOG_POSEIDON_DEBUG("Dispatching control message: ", req);
					session->onControl(static_cast<ControlCode>(req.messageId), req.statusCode, STD_MOVE(req.reason));
				} else {
					LOG_POSEIDON_DEBUG("Dispatching message: messageId = ", m_messageId, ", payload size = ", m_payload.size());
					session->onRequest(m_messageId, m_payload);
				}
				session->setTimeout(MainConfig::getConfigFile().get<boost::uint64_t>("cbpp_keep_alive_timeout", 0));
			} catch(TryAgainLater &){
				throw;
			} catch(Exception &e){
				LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "Cbpp::Exception thrown: message id = ",
					m_messageId, ", statusCode = ", e.statusCode(), ", what = ", e.what());
				try {
					session->sendError(m_messageId, e.statusCode(), e.what(), false); // 不关闭连接。
				} catch(...){
					session->forceShutdown();
				}
			}
		}
	};

	class Session::ErrorJob : public SessionJobBase {
	private:
		const unsigned m_messageId;
		const StatusCode m_statusCode;
		const std::string m_reason;

	public:
		ErrorJob(const boost::shared_ptr<Session> &session, unsigned messageId, StatusCode statusCode, std::string reason)
			: SessionJobBase(session)
			, m_messageId(messageId), m_statusCode(statusCode), m_reason(STD_MOVE(reason))
		{
		}

	protected:
		void perform(const boost::shared_ptr<Session> &session) const OVERRIDE {
			PROFILE_ME;

			session->sendError(m_messageId, m_statusCode, m_reason, true);
		}
	};

	Session::Session(UniqueFile socket)
		: TcpSessionBase(STD_MOVE(socket))
		, m_sizeTotal(0), m_sizeExpecting(2), m_state(S_PAYLOAD_LEN)
	{
	}
	Session::~Session(){
		if(m_state != S_PAYLOAD_LEN){
			LOG_POSEIDON_WARNING("Now that this session is to be destroyed, a premature request has to be discarded.");
		}
	}

	void Session::onReadAvail(const void *data, std::size_t size){
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
					enqueueJob(boost::make_shared<RequestJob>(
						virtualSharedFromThis<Session>(), m_messageId, m_received.cut(m_payloadLen)));

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
			LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "Cbpp::Exception thrown while parsing data, message id = ", m_messageId,
				", statusCode = ", static_cast<int>(e.statusCode()), ", what = ", e.what());
			try {
				enqueueJob(boost::make_shared<ErrorJob>(
					virtualSharedFromThis<Session>(), m_messageId, e.statusCode(), e.what()));
				setPreservedOnReadHup(true); // noexcept
				shutdown();
			} catch(...){
				forceShutdown();
			}
		} catch(std::exception &e){
			LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "std::exception thrown while parsing data, message id = ", m_messageId,
				", what = ", e.what());
			try {
				enqueueJob(boost::make_shared<ErrorJob>(
					virtualSharedFromThis<Session>(), m_messageId, ST_INTERNAL_ERROR, std::string()));
				setPreservedOnReadHup(true); // noexcept
				shutdown();
			} catch(...){
				forceShutdown();
			}
		}
	}

	void Session::onControl(ControlCode controlCode, StatusCode statusCode, std::string reason){
		PROFILE_ME;

		switch(controlCode){
		case CTL_HEARTBEAT:
			LOG_POSEIDON_TRACE("Received heartbeat from ", getRemoteInfo());
			break;

		default:
			LOG_POSEIDON_WARNING("Unknown control code: ", controlCode);
			send(ErrorMessage(static_cast<boost::uint16_t>(controlCode), statusCode, STD_MOVE(reason)), true);
			break;
		}
	}

	bool Session::send(boost::uint16_t messageId, StreamBuffer payload, bool fin){
		PROFILE_ME;

		LOG_POSEIDON_DEBUG("Sending frame: messageId = ", messageId, ", size = ", payload.size(), ", fin = ", fin);
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
		return TcpSessionBase::send(STD_MOVE(frame), fin);
	}

	bool Session::sendError(boost::uint16_t messageId, StatusCode statusCode, std::string reason, bool fin){
		return send(ErrorMessage(messageId, static_cast<int>(statusCode), STD_MOVE(reason)), fin);
	}
}

}
