// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "session.hpp"
#include "exception.hpp"
#include "control_codes.hpp"
#include "error_message.hpp"
#include "../log.hpp"
#include "../exception.hpp"
#include "../singletons/cbpp_servlet_depository.hpp"
#include "../job_base.hpp"
#include "../profiler.hpp"

namespace Poseidon {

namespace Cbpp {
	namespace {
		class RequestJob : public JobBase {
		private:
			const boost::weak_ptr<Session> m_session;
			const unsigned m_messageId;
			const StreamBuffer m_payload;

		public:
			RequestJob(boost::weak_ptr<Session> session, unsigned messageId, StreamBuffer payload)
				: m_session(STD_MOVE(session)), m_messageId(messageId), m_payload(STD_MOVE(payload))
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
					if(m_messageId == ErrorMessage::ID){
						AUTO(payload, m_payload);
						ErrorMessage packet(payload);
						LOG_POSEIDON_DEBUG("Received error packet: message id = ", packet.messageId,
							", statusCode = ", packet.statusCode, ", reason = ", packet.reason);

						switch(static_cast<ControlCode>(packet.messageId)){
						case CTL_HEARTBEAT:
							LOG_POSEIDON_TRACE("Received heartbeat from ", session->getRemoteInfo());
							break;

						default:
							LOG_POSEIDON_WARNING("Unknown control code: ", packet.messageId);
							session->send(ErrorMessage::ID, StreamBuffer(packet), true);
							break;
						}
					} else {
						const AUTO(category, session->getCategory());
						const AUTO(servlet, ServletDepository::get(category, m_messageId));
						if(!servlet){
							LOG_POSEIDON_WARNING("No servlet in category ", category, " matches message ", m_messageId);
							DEBUG_THROW(Exception, ST_NOT_FOUND, SharedNts::observe("Unknown message"));
						}

						LOG_POSEIDON_DEBUG("Dispatching packet: message = ", m_messageId,
							", payload size = ", m_payload.size());
						(*servlet)(session, m_payload);
					}
					session->setTimeout(ServletDepository::getKeepAliveTimeout());
				} catch(TryAgainLater &){
					throw;
				} catch(Exception &e){
					LOG_POSEIDON_ERROR("Exception thrown in  servlet, message id = ",
						m_messageId, ", statusCode = ", e.statusCode(), ", what = ", e.what());
					session->sendError(m_messageId, e.statusCode(), e.what(), false); // 不关闭连接。
					throw;
				} catch(...){
					LOG_POSEIDON_ERROR("Forwarding exception... message id = ", m_messageId);
					session->sendError(m_messageId, ST_INTERNAL_ERROR, true); // 关闭连接。
					throw;
				}
			}
		};
	}

	Session::Session(std::size_t category, UniqueFile socket)
		: TcpSessionBase(STD_MOVE(socket))
		, m_category(category)
		, m_payloadLen((boost::uint64_t)-1), m_messageId(0)
	{
	}
	Session::~Session(){
		if(m_payloadLen != (boost::uint64_t)-1){
			LOG_POSEIDON_WARNING(
				"Now that this session is to be destroyed, a premature request has to be discarded.");
		}
	}

	void Session::onReadAvail(const void *data, std::size_t size){
		PROFILE_ME;

		try {
			m_payload.put(data, size);
			for(;;){
				if(m_payloadLen == (boost::uint64_t)-1){
					boost::uint16_t messageId;
					boost::uint64_t payloadLen;
					if(!MessageBase::decodeHeader(messageId, payloadLen, m_payload)){
						break;
					}
					m_messageId = messageId;
					m_payloadLen = payloadLen;
					LOG_POSEIDON_DEBUG("Message id = ", m_messageId, ", len = ", m_payloadLen);

					const std::size_t maxRequestLength = ServletDepository::getMaxRequestLength();
					if((unsigned)m_payloadLen >= maxRequestLength){
						LOG_POSEIDON_WARNING("Request too large: size = ", m_payloadLen, ", max = ", maxRequestLength);
						DEBUG_THROW(Exception, ST_REQUEST_TOO_LARGE, SharedNts::observe("Request too large"));
					}
				}
				if(m_payload.size() < (unsigned)m_payloadLen){
					break;
				}
				enqueueJob(boost::make_shared<RequestJob>(
					virtualWeakFromThis<Session>(), m_messageId, m_payload.cut(m_payloadLen)));
				m_payloadLen = (boost::uint64_t)-1;
				m_messageId = 0;
			}
		} catch(Exception &e){
			LOG_POSEIDON_ERROR("Exception thrown while parsing data, message id = ", m_messageId,
				", statusCode = ", static_cast<int>(e.statusCode()), ", what = ", e.what());
			sendError(m_messageId, e.statusCode(), e.what(), true);
			throw;
		} catch(...){
			LOG_POSEIDON_ERROR("Forwarding exception... message id = ", m_messageId);
			sendError(m_messageId, ST_INTERNAL_ERROR, true);
			throw;
		}
	}

	bool Session::send(boost::uint16_t messageId, StreamBuffer contents, bool fin){
		LOG_POSEIDON_DEBUG("Sending data: message id = ", messageId,
			", content length = ", contents.size(), ", fin = ", std::boolalpha, fin);
		StreamBuffer data;
		MessageBase::encodeHeader(data, messageId, contents.size());
		data.splice(contents);
		return TcpSessionBase::send(STD_MOVE(data), fin);
	}

	bool Session::sendError(boost::uint16_t messageId, StatusCode statusCode,
		std::string reason, bool fin)
	{
		return send(ErrorMessage::ID, StreamBuffer(ErrorMessage(
			messageId, static_cast<int>(statusCode), STD_MOVE(reason))), fin);
	}
}

}
