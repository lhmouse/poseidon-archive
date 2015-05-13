// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "session.hpp"
#include "exception.hpp"
#include "../http/low_level_session.hpp"
#include "../optional_map.hpp"
#include "../singletons/main_config.hpp"
#include "../log.hpp"
#include "../job_base.hpp"
#include "../profiler.hpp"

namespace Poseidon {

namespace WebSocket {
	namespace {
		class SessionJobBase : public JobBase {
		private:
			const boost::weak_ptr<Http::LowLevelSession> m_parent;
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
				LOG_POSEIDON_DEBUG("Dispatching request: payload size = ", m_payload.size());
				session->onRequest(m_opcode, m_payload);
				session->setTimeout(MainConfig::get().get<boost::uint64_t>("websocket_keep_alive_timeout", 30000));
			} catch(TryAgainLater &){
				throw;
			} catch(Exception &e){
				LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "WebSocket::Exception thrown: statusCode = ", e.statusCode(), ", what = ", e.what());
				session->shutdown(e.statusCode(), StreamBuffer(e.what()));
			} catch(std::exception &e){
				LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "std::exception thrown: what = ", e.what());
				session->shutdown(ST_INTERNAL_ERROR, StreamBuffer(e.what()));
			}
		}
	};

	class Session::ErrorJob : public SessionJobBase {
	private:
		const TcpSessionBase::DelayedShutdownGuard m_guard;

		const StatusCode m_statusCode;
		const StreamBuffer m_additional;

	public:
		ErrorJob(const boost::shared_ptr<Session> &session, StatusCode statusCode, StreamBuffer additional)
			: SessionJobBase(session)
			, m_guard(session->getSafeParent())
			, m_statusCode(statusCode), m_additional(STD_MOVE(additional))
		{
		}

	protected:
		void perform(const boost::shared_ptr<Session> &session) const OVERRIDE {
			PROFILE_ME;

			session->shutdown(m_statusCode, m_additional);
		}
	};

	Session::Session(const boost::shared_ptr<Http::LowLevelSession> &parent)
		: LowLevelSession(parent)
	{
	}
	Session::~Session(){
	}

	void Session::onLowLevelRequest(OpCode opcode, StreamBuffer payload){
		PROFILE_ME;

		enqueueJob(boost::make_shared<RequestJob>(
			virtualSharedFromThis<Session>(), opcode, STD_MOVE(payload)));
	}
	void Session::onLowLevelError(StatusCode statusCode, const char *reason){
		PROFILE_ME;

		enqueueJob(boost::make_shared<ErrorJob>(
			virtualSharedFromThis<Session>(), statusCode, StreamBuffer(reason)));
	}
}

}
