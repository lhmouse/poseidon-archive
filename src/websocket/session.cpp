// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "session.hpp"
#include "exception.hpp"
#include "writer.hpp"
#include "../http/session.hpp"
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
				} catch(Exception &e){
					LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO,
						"WebSocket::Exception thrown: statusCode = ", e.statusCode(), ", what = ", e.what());
					session->shutdown(e.statusCode(), StreamBuffer(e.what()));
				} catch(std::exception &e){
					LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO,
						"std::exception thrown: what = ", e.what());
					session->forceShutdown();
					throw;
				} catch(...){
					LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO,
						"Unknown exception thrown.");
					session->forceShutdown();
					throw;
				}
			}
		};

		class SessionWriter : public Writer {
		private:
			Http::UpgradedSessionBase *m_session;

		public:
			explicit SessionWriter(Http::UpgradedSessionBase &session)
				: m_session(&session)
			{
			}

		protected:
			long onEncodedDataAvail(StreamBuffer encoded) OVERRIDE {
				PROFILE_ME;

				return m_session->send(STD_MOVE(encoded));
			}
		};
	}

	class Session::DataMessageJob : public SessionJobBase {
	private:
		const OpCode m_opcode;
		const StreamBuffer m_payload;

	public:
		DataMessageJob(const boost::shared_ptr<Session> &session, OpCode opcode, StreamBuffer payload)
			: SessionJobBase(session)
			, m_opcode(opcode), m_payload(STD_MOVE(payload))
		{
		}

	protected:
		void perform(const boost::shared_ptr<Session> &session) const OVERRIDE {
			PROFILE_ME;

			LOG_POSEIDON_DEBUG("Dispatching data message: opcode = ", m_opcode, ", payloadSize = ", m_payload.size());
			session->onSyncDataMessage(m_opcode, m_payload);

			const AUTO(keepAliveTimeout, MainConfig::get().get<boost::uint64_t>("websocket_keep_alive_timeout", 30000));
			session->setTimeout(keepAliveTimeout);
		}
	};

	class Session::ControlMessageJob : public SessionJobBase {
	private:
		const OpCode m_opcode;
		const StreamBuffer m_payload;

	public:
		ControlMessageJob(const boost::shared_ptr<Session> &session, OpCode opcode, StreamBuffer payload)
			: SessionJobBase(session)
			, m_opcode(opcode), m_payload(STD_MOVE(payload))
		{
		}

	protected:
		void perform(const boost::shared_ptr<Session> &session) const OVERRIDE {
			PROFILE_ME;

			LOG_POSEIDON_DEBUG("Dispatching control message: opcode = ", m_opcode, ", payloadSize = ", m_payload.size());
			session->onSyncControlMessage(m_opcode, m_payload);

			const AUTO(keepAliveTimeout, MainConfig::get().get<boost::uint64_t>("websocket_keep_alive_timeout", 30000));
			session->setTimeout(keepAliveTimeout);
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

	Session::Session(const boost::shared_ptr<Http::Session> &parent)
		: Http::UpgradedSessionBase(parent)
		, m_opcode(OP_INVALID_OPCODE)
	{
	}
	Session::~Session(){
	}

	void Session::onReadAvail(const void *data, std::size_t size){
		PROFILE_ME;

		try {
			const AUTO(maxRequestLength, MainConfig::get().get<boost::uint64_t>("websocket_max_request_length", 16384));
			if(m_sizeTotal > maxRequestLength){
				DEBUG_THROW(Exception, ST_MESSAGE_TOO_LARGE, SSLIT("Message too large"));
			}

			Reader::putEncodedData(StreamBuffer(data, size));
		} catch(Exception &e){
			LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO,
				"WebSocket::Exception thrown in WebSocket parser: statusCode = ", e.statusCode(), ", what = ", e.what());
			enqueueJob(boost::make_shared<ErrorJob>(
				virtualSharedFromThis<Session>(), e.statusCode(), StreamBuffer(e.what())));
			shutdownRead();
			shutdownWrite();
		}
	}

	void Session::onDataMessageHeader(OpCode opcode){
		PROFILE_ME;

		m_opcode = opcode;
		m_payload.clear();
	}
	void Session::onDataMessagePayload(boost::uint64_t /* wholeOffset */, StreamBuffer payload){
		PROFILE_ME;

		m_payload.splice(payload);
	}
	bool Session::onDataMessageEnd(boost::uint64_t /* wholeSize */){
		PROFILE_ME;

		enqueueJob(boost::make_shared<DataMessageJob>(
			virtualSharedFromThis<Session>(), m_opcode, STD_MOVE(m_payload)));
		m_sizeTotal = 0;

		return true;
	}

	bool Session::onControlMessage(OpCode opcode, StreamBuffer payload){
		PROFILE_ME;

		enqueueJob(boost::make_shared<ControlMessageJob>(
			virtualSharedFromThis<Session>(), opcode, STD_MOVE(payload)));

		return true;
	}

	void Session::onSyncControlMessage(OpCode opcode, const StreamBuffer &payload){
		PROFILE_ME;
		LOG_POSEIDON_DEBUG("Control frame, opcode = ", m_opcode);

		const AUTO(parent, getSafeParent());

		SessionWriter writer(*this);

		switch(opcode){
		case OP_CLOSE:
			LOG_POSEIDON_INFO("Received close frame from ", parent->getRemoteInfo());
			writer.putCloseMessage(ST_NORMAL_CLOSURE, VAL_INIT);
			shutdownRead();
			shutdownWrite();
			break;

		case OP_PING:
			LOG_POSEIDON_INFO("Received ping frame from ", parent->getRemoteInfo());
			writer.putMessage(OP_PONG, false, payload);
			break;

		case OP_PONG:
			LOG_POSEIDON_INFO("Received pong frame from ", parent->getRemoteInfo());
			break;

		default:
			DEBUG_THROW(Exception, ST_PROTOCOL_ERROR, SSLIT("Invalid opcode"));
			break;
		}
	}

	bool Session::send(StreamBuffer payload, bool binary, bool masked){
		PROFILE_ME;

		SessionWriter writer(*this);
		return writer.putMessage(binary ? OP_DATA_BIN : OP_DATA_TEXT, masked, STD_MOVE(payload));
	}

	bool Session::shutdown(StatusCode statusCode, StreamBuffer additional) NOEXCEPT {
		PROFILE_ME;

		try {
			SessionWriter writer(*this);
			writer.putCloseMessage(statusCode, STD_MOVE(additional));
			shutdownRead();
			return shutdownWrite();
		} catch(std::exception &e){
			LOG_POSEIDON_WARNING("std::exception thrown: what = ", e.what());
			forceShutdown();
			return false;
		}
	}
}

}
