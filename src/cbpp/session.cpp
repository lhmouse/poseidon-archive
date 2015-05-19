// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "session.hpp"
#include "exception.hpp"
#include "control_message.hpp"
#include "../singletons/main_config.hpp"
#include "../log.hpp"
#include "../profiler.hpp"
#include "../job_base.hpp"

namespace Poseidon {

namespace Cbpp {
	class Session::SyncJobBase : public JobBase {
	private:
		const boost::weak_ptr<Session> m_session;

	protected:
		explicit SyncJobBase(const boost::shared_ptr<Session> &session)
			: m_session(session)
		{
		}

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
			} catch(Exception &e){
				LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO,
					"Cbpp::Exception thrown: statusCode = ", e.statusCode(), ", what = ", e.what());
				try {
					session->sendError(ControlMessage::ID, e.statusCode(), e.what());
					session->shutdownRead();
					session->shutdownWrite();
				} catch(...){
					session->forceShutdown();
				}
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

	protected:
		virtual void perform(const boost::shared_ptr<Session> &session) const = 0;
	};

	class Session::DataMessageJob : public SyncJobBase {
	private:
		const boost::uint16_t m_messageId;
		const StreamBuffer m_payload;

	public:
		DataMessageJob(const boost::shared_ptr<Session> &session,
			boost::uint16_t messageId, StreamBuffer payload)
			: SyncJobBase(session)
			, m_messageId(messageId), m_payload(STD_MOVE(payload))
		{
		}

	protected:
		void perform(const boost::shared_ptr<Session> &session) const OVERRIDE {
			PROFILE_ME;

			LOG_POSEIDON_DEBUG("Dispatching message: messageId = ", m_messageId, ", payloadLen = ", m_payload.size());
			session->onSyncDataMessage(m_messageId, m_payload);

			const AUTO(keepAliveTimeout, MainConfig::get<boost::uint64_t>("cbpp_keep_alive_timeout", 30000));
			session->setTimeout(keepAliveTimeout);
		}
	};

	class Session::ControlMessageJob : public SyncJobBase {
	private:
		const ControlCode m_controlCode;
		const boost::int64_t m_vintParam;
		const std::string m_stringParam;

	public:
		ControlMessageJob(const boost::shared_ptr<Session> &session,
			ControlCode controlCode, boost::int64_t vintParam, std::string stringParam)
			: SyncJobBase(session)
			, m_controlCode(controlCode), m_vintParam(vintParam), m_stringParam(stringParam)
		{
		}

	protected:
		void perform(const boost::shared_ptr<Session> &session) const OVERRIDE {
			PROFILE_ME;

			LOG_POSEIDON_DEBUG("Dispatching control message: controlCode = ", m_controlCode,
				", vintParam = ", m_vintParam, ", stringParam = ", m_stringParam);
			session->onSyncControlMessage(m_controlCode, m_vintParam, m_stringParam);

			const AUTO(keepAliveTimeout, MainConfig::get<boost::uint64_t>("cbpp_keep_alive_timeout", 30000));
			session->setTimeout(keepAliveTimeout);
		}
	};

	class Session::ErrorJob : public SyncJobBase {
	private:
		const TcpSessionBase::DelayedShutdownGuard m_guard;

		const boost::uint16_t m_messageId;
		const StatusCode m_statusCode;
		const std::string m_reason;

	public:
		ErrorJob(const boost::shared_ptr<Session> &session,
			boost::uint16_t messageId, StatusCode statusCode, std::string reason)
			: SyncJobBase(session)
			, m_guard(session)
			, m_messageId(messageId), m_statusCode(statusCode), m_reason(STD_MOVE(reason))
		{
		}

	protected:
		void perform(const boost::shared_ptr<Session> &session) const OVERRIDE {
			PROFILE_ME;

			session->sendError(m_messageId, m_statusCode, m_reason);
		}
	};

	Session::Session(UniqueFile socket)
		: TcpSessionBase(STD_MOVE(socket))
		, m_sizeTotal(0), m_messageId(0), m_payload()
	{
	}
	Session::~Session(){
	}

	void Session::onReadAvail(const void *data, std::size_t size){
		PROFILE_ME;

		try {
			m_sizeTotal += size;
			const AUTO(maxRequestLength, MainConfig::get<boost::uint64_t>("cbpp_max_request_length", 16384));
			if(m_sizeTotal > maxRequestLength){
				DEBUG_THROW(Exception, ST_REQUEST_TOO_LARGE);
			}

			Reader::putEncodedData(StreamBuffer(data, size));
		} catch(Exception &e){
			LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO,
				"Cbpp::Exception thrown: statusCode = ", e.statusCode(), ", what = ", e.what());
			enqueueJob(boost::make_shared<ErrorJob>(
				virtualSharedFromThis<Session>(), Reader::getMessageId(), e.statusCode(), e.what()));
		}
	}
	void Session::onDataMessageHeader(boost::uint16_t messageId, boost::uint64_t /* payloadSize */){
		PROFILE_ME;

		m_messageId = messageId;
		m_payload.clear();
	}
	void Session::onDataMessagePayload(boost::uint64_t /* payloadOffset */, StreamBuffer payload){
		PROFILE_ME;

		m_payload.splice(payload);
	}
	bool Session::onDataMessageEnd(boost::uint64_t /* payloadSize */){
		PROFILE_ME;

		enqueueJob(boost::make_shared<DataMessageJob>(
			virtualSharedFromThis<Session>(), m_messageId, STD_MOVE(m_payload)));
		m_sizeTotal = 0;

		return true;
	}

	bool Session::onControlMessage(ControlCode controlCode, boost::int64_t vintParam, std::string stringParam){
		PROFILE_ME;

		enqueueJob(boost::make_shared<ControlMessageJob>(
			virtualSharedFromThis<Session>(), controlCode, vintParam, STD_MOVE(stringParam)));
		m_sizeTotal = 0;
		m_messageId = 0;
		m_payload.clear();

		return true;
	}

	long Session::onEncodedDataAvail(StreamBuffer encoded){
		PROFILE_ME;

		return TcpSessionBase::send(STD_MOVE(encoded));
	}

	void Session::onSyncControlMessage(ControlCode controlCode, boost::int64_t vintParam, const std::string &stringParam){
		PROFILE_ME;

		switch(controlCode){
		case CTL_HEARTBEAT:
			LOG_POSEIDON_TRACE("Received heartbeat from ", getRemoteInfo());
			break;

		default:
			LOG_POSEIDON_WARNING("Unknown control code: ", controlCode);
			send(ControlMessage(controlCode, vintParam, stringParam));
			shutdownRead();
			shutdownWrite();
			break;
		}
	}

	bool Session::send(boost::uint16_t messageId, StreamBuffer payload){
		PROFILE_ME;

		return Writer::putDataMessage(messageId, STD_MOVE(payload));
	}
	bool Session::sendError(boost::uint16_t messageId, StatusCode statusCode, std::string reason){
		PROFILE_ME;

		return Writer::putControlMessage(messageId, statusCode, STD_MOVE(reason));
	}
}

}
