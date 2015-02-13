// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "session.hpp"
#include "exception.hpp"
#include "control_code.hpp"
#include "error_message.hpp"
#include "../log.hpp"
#include "../exception.hpp"
#include "../singletons/cbpp_servlet_depository.hpp"
#include "../job_base.hpp"
#include "../profiler.hpp"

namespace Poseidon {

namespace {
	class CbppRequestJob : public JobBase {
	private:
		const boost::weak_ptr<CbppSession> m_session;
		const unsigned m_messageId;

		StreamBuffer m_payload;

	public:
		CbppRequestJob(boost::weak_ptr<CbppSession> session,
			unsigned messageId, StreamBuffer payload)
			: m_session(STD_MOVE(session)), m_messageId(messageId)
			, m_payload(STD_MOVE(payload))
		{
		}

	protected:
		void perform(){
			PROFILE_ME;

			const boost::shared_ptr<CbppSession> session(m_session);
			try {
				if(m_messageId == CbppErrorMessage::ID){
					CbppErrorMessage packet(m_payload);
					LOG_POSEIDON_DEBUG("Received error packet: message id = ", packet.messageId,
						", status = ", packet.status, ", reason = ", packet.reason);

					switch(static_cast<CbppControlCode>(packet.messageId)){
					case CBPP_CTL_HEARTBEAT:
						LOG_POSEIDON_TRACE("Received heartbeat from ", session->getRemoteInfo());
						break;

					default:
						LOG_POSEIDON_WARNING("Unknown control code: ", packet.messageId);
						session->send(CbppErrorMessage::ID, StreamBuffer(packet), true);
						break;
					}
				} else {
					const AUTO(category, session->getCategory());
					const AUTO(servlet, CbppServletDepository::getServlet(category, m_messageId));
					if(!servlet){
						LOG_POSEIDON_WARNING(
							"No servlet in category ", category, " matches message ", m_messageId);
						DEBUG_THROW(CbppMessageException, CBPP_NOT_FOUND,
							SharedNts::observe("Unknown message"));
					}

					LOG_POSEIDON_DEBUG("Dispatching packet: message = ", m_messageId,
						", payload size = ", m_payload.size());
					(*servlet)(session, STD_MOVE(m_payload));
				}
				session->setTimeout(CbppServletDepository::getKeepAliveTimeout());
			} catch(CbppMessageException &e){
				LOG_POSEIDON_ERROR("CbppMessageException thrown in CBPP servlet, message id = ",
					m_messageId, ", status = ", e.status(), ", what = ", e.what());
				session->sendError(m_messageId, e.status(), e.what(), false); // 不关闭连接。
				throw;
			} catch(...){
				LOG_POSEIDON_ERROR("Forwarding exception... message id = ", m_messageId);
				session->sendError(m_messageId, CBPP_INTERNAL_ERROR, true); // 关闭连接。
				throw;
			}
		}
	};
}

CbppSession::CbppSession(std::size_t category, UniqueFile socket)
	: TcpSessionBase(STD_MOVE(socket))
	, m_category(category)
	, m_payloadLen((boost::uint64_t)-1), m_messageId(0)
{
}
CbppSession::~CbppSession(){
	if(m_payloadLen != (boost::uint64_t)-1){
		LOG_POSEIDON_WARNING(
			"Now that this session is to be destroyed, a premature request has to be discarded.");
	}
}

void CbppSession::onReadAvail(const void *data, std::size_t size){
	PROFILE_ME;

	try {
		m_payload.put(data, size);
		for(;;){
			if(m_payloadLen == (boost::uint64_t)-1){
				boost::uint16_t messageId;
				boost::uint64_t payloadLen;
				if(!CbppMessageBase::decodeHeader(messageId, payloadLen, m_payload)){
					break;
				}
				m_messageId = messageId;
				m_payloadLen = payloadLen;
				LOG_POSEIDON_DEBUG("Message id = ", m_messageId, ", len = ", m_payloadLen);

				const std::size_t maxRequestLength = CbppServletDepository::getMaxRequestLength();
				if((unsigned)m_payloadLen >= maxRequestLength){
					LOG_POSEIDON_WARNING(
						"Request too large: size = ", m_payloadLen, ", max = ", maxRequestLength);
					DEBUG_THROW(CbppMessageException, CBPP_REQUEST_TOO_LARGE,
						SharedNts::observe("Request too large"));
				}
			}
			if(m_payload.size() < (unsigned)m_payloadLen){
				break;
			}
			pendJob(boost::make_shared<CbppRequestJob>(virtualWeakFromThis<CbppSession>(),
				m_messageId, m_payload.cut(m_payloadLen)));
			m_payloadLen = (boost::uint64_t)-1;
			m_messageId = 0;
		}
	} catch(CbppMessageException &e){
		LOG_POSEIDON_ERROR(
			"CbppMessageException thrown while parsing data, message id = ", m_messageId,
			", status = ", static_cast<int>(e.status()), ", what = ", e.what());
		sendError(m_messageId, e.status(), e.what(), true);
		throw;
	} catch(...){
		LOG_POSEIDON_ERROR("Forwarding exception... message id = ", m_messageId);
		sendError(m_messageId, CBPP_INTERNAL_ERROR, true);
		throw;
	}
}

bool CbppSession::send(boost::uint16_t messageId, StreamBuffer contents, bool fin){
	LOG_POSEIDON_DEBUG("Sending data: message id = ", messageId,
		", content length = ", contents.size(), ", fin = ", std::boolalpha, fin);
	StreamBuffer data;
	CbppMessageBase::encodeHeader(data, messageId, contents.size());
	data.splice(contents);
	return TcpSessionBase::send(STD_MOVE(data), fin);
}

bool CbppSession::sendError(boost::uint16_t messageId, CbppStatus status,
	std::string reason, bool fin)
{
	return send(CbppErrorMessage::ID, StreamBuffer(CbppErrorMessage(
		messageId, static_cast<int>(status), STD_MOVE(reason))), fin);
}

}
