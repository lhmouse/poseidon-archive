// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "session.hpp"
#include "exception.hpp"
#include "control_message.hpp"
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
				LOG_POSEIDON_DEBUG("Dispatching message: messageId = ", m_messageId, ", payloadLen = ", m_payload.size());
				session->onRequest(m_messageId, m_payload);

				session->setTimeout(MainConfig::getConfigFile().get<boost::uint64_t>("cbpp_keep_alive_timeout", 30000));
			} catch(TryAgainLater &){
				throw;
			} catch(Exception &e){
				LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO,
					"Cbpp::Exception thrown: messageId = ", m_messageId, ", statusCode = ", e.statusCode(), ", what = ", e.what());
				session->sendControl(m_messageId, e.statusCode(), e.what());
				session->shutdownRead();
				session->shutdownWrite();
			} catch(std::exception &e){
				LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO,
					"std::exception thrown: messageId = ", m_messageId, ", what = ", e.what());
				session->sendControl(m_messageId, ST_INTERNAL_ERROR, e.what());
				session->shutdownRead();
				session->shutdownWrite();
			}
		}
	};

	class Session::ControlJob : public SessionJobBase {
	private:
		const ControlCode m_controlCode;
		const StatusCode m_statusCode;
		const std::string m_reason;

	public:
		ControlJob(const boost::shared_ptr<Session> &session,
			ControlCode controlCode, StatusCode statusCode, std::string reason)
			: SessionJobBase(session)
			, m_controlCode(controlCode), m_statusCode(statusCode), m_reason(STD_MOVE(reason))
		{
		}

	protected:
		void perform(const boost::shared_ptr<Session> &session) const OVERRIDE {
			PROFILE_ME;

			try {
				LOG_POSEIDON_DEBUG("Dispatching control message: controlCode = ", m_controlCode,
					", statusCode = ", m_statusCode, ", reason = ", m_reason);
				session->onControl(m_controlCode, m_statusCode, m_reason);

				session->setTimeout(MainConfig::getConfigFile().get<boost::uint64_t>("cbpp_keep_alive_timeout", 30000));
			} catch(Exception &e){
				LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO,
					"Cbpp::Exception thrown: statusCode = ", e.statusCode(), ", what = ", e.what());
				session->sendControl(ControlMessage::ID, e.statusCode(), e.what());
				session->shutdownRead();
				session->shutdownWrite();
			}
		}
	};

	class Session::ErrorJob : public SessionJobBase {
	private:
		const TcpSessionBase::DelayedShutdownGuard m_guard;

		const unsigned m_messageId;
		const StatusCode m_statusCode;
		const std::string m_reason;

	public:
		ErrorJob(const boost::shared_ptr<Session> &session, unsigned messageId, StatusCode statusCode, std::string reason)
			: SessionJobBase(session)
			, m_guard(session)
			, m_messageId(messageId), m_statusCode(statusCode), m_reason(STD_MOVE(reason))
		{
		}

	protected:
		void perform(const boost::shared_ptr<Session> &session) const OVERRIDE {
			PROFILE_ME;

			session->sendControl(m_messageId, m_statusCode, m_reason);
		}
	};

	Session::Session(UniqueFile socket)
		: LowLevelSession(STD_MOVE(socket))
	{
	}
	Session::~Session(){
	}

	void Session::onLowLevelRequest(boost::uint16_t messageId, StreamBuffer payload){
		PROFILE_ME;

		enqueueJob(boost::make_shared<RequestJob>(
			virtualSharedFromThis<Session>(), messageId, STD_MOVE(payload)));
	}
	void Session::onLowLevelControl(ControlCode controlCode, StatusCode statusCode, std::string reason){
		PROFILE_ME;

		enqueueJob(boost::make_shared<ControlJob>(
			virtualSharedFromThis<Session>(), controlCode, statusCode, STD_MOVE(reason)));
	}

	void Session::onLowLevelError(unsigned messageId, StatusCode statusCode, const char *reason){
		PROFILE_ME;

		enqueueJob(boost::make_shared<ErrorJob>(
			virtualSharedFromThis<Session>(), messageId, statusCode, std::string(reason)));

		shutdownRead();
		shutdownWrite();
	}

	void Session::onControl(ControlCode controlCode, StatusCode statusCode, const std::string &reason){
		PROFILE_ME;

		switch(controlCode){
		case CTL_HEARTBEAT:
			LOG_POSEIDON_TRACE("Received heartbeat from ", getRemoteInfo());
			break;

		default:
			LOG_POSEIDON_WARNING("Unknown control code: ", controlCode);
			send(ControlMessage(static_cast<boost::uint16_t>(controlCode), statusCode, reason));
			shutdownRead();
			shutdownWrite();
			break;
		}
	}
}

}
