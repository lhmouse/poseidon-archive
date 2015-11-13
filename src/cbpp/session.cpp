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
#include "../time.hpp"

namespace Poseidon {

namespace Cbpp {
	class Session::SyncJobBase : public JobBase {
	private:
		const TcpSessionBase::DelayedShutdownGuard m_guard;
		const boost::weak_ptr<Session> m_session;

	protected:
		explicit SyncJobBase(const boost::shared_ptr<Session> &session)
			: m_guard(session), m_session(session)
		{
		}

	private:
		boost::weak_ptr<const void> get_category() const FINAL {
			return m_session;
		}
		void perform() FINAL {
			PROFILE_ME;

			const AUTO(session, m_session.lock());
			if(!session){
				return;
			}

			try {
				really_perform(session);
			} catch(Exception &e){
				LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO,
					"Cbpp::Exception thrown: status_code = ", e.status_code(), ", what = ", e.what());
				try {
					session->send_error(ControlMessage::ID, e.status_code(), e.what());
					session->shutdown_read();
					session->shutdown_write();
				} catch(...){
					session->force_shutdown();
				}
				throw;
			} catch(std::exception &e){
				LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "std::exception thrown: what = ", e.what());
				session->force_shutdown();
				throw;
			} catch(...){
				LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "Unknown exception thrown.");
				session->force_shutdown();
				throw;
			}
		}

	protected:
		virtual void really_perform(const boost::shared_ptr<Session> &session) = 0;
	};

	class Session::DataMessageJob : public Session::SyncJobBase {
	private:
		boost::uint16_t m_message_id;
		StreamBuffer m_payload;

	public:
		DataMessageJob(const boost::shared_ptr<Session> &session,
			boost::uint16_t message_id, StreamBuffer payload)
			: SyncJobBase(session)
			, m_message_id(message_id), m_payload(STD_MOVE(payload))
		{
		}

	protected:
		void really_perform(const boost::shared_ptr<Session> &session) OVERRIDE {
			PROFILE_ME;

			LOG_POSEIDON_DEBUG("Dispatching message: message_id = ", m_message_id, ", payload_len = ", m_payload.size());
			session->on_sync_data_message(m_message_id, STD_MOVE(m_payload));

			const AUTO(keep_alive_timeout, MainConfig::get<boost::uint64_t>("cbpp_keep_alive_timeout", 30000));
			session->set_timeout(keep_alive_timeout);
		}
	};

	class Session::ControlMessageJob : public Session::SyncJobBase {
	private:
		ControlCode m_control_code;
		boost::int64_t m_vint_param;
		std::string m_string_param;

	public:
		ControlMessageJob(const boost::shared_ptr<Session> &session,
			ControlCode control_code, boost::int64_t vint_param, std::string string_param)
			: SyncJobBase(session)
			, m_control_code(control_code), m_vint_param(vint_param), m_string_param(STD_MOVE(string_param))
		{
		}

	protected:
		void really_perform(const boost::shared_ptr<Session> &session) OVERRIDE {
			PROFILE_ME;

			LOG_POSEIDON_DEBUG("Dispatching control message: control_code = ", m_control_code,
				", vint_param = ", m_vint_param, ", string_param = ", m_string_param);
			session->on_sync_control_message(m_control_code, m_vint_param, STD_MOVE(m_string_param));

			const AUTO(keep_alive_timeout, MainConfig::get<boost::uint64_t>("cbpp_keep_alive_timeout", 30000));
			session->set_timeout(keep_alive_timeout);
		}
	};

	class Session::ErrorJob : public Session::SyncJobBase {
	private:
		boost::uint16_t m_message_id;
		StatusCode m_status_code;
		std::string m_reason;

	public:
		ErrorJob(const boost::shared_ptr<Session> &session,
			boost::uint16_t message_id, StatusCode status_code, std::string reason)
			: SyncJobBase(session)
			, m_message_id(message_id), m_status_code(status_code), m_reason(STD_MOVE(reason))
		{
		}

	protected:
		void really_perform(const boost::shared_ptr<Session> &session) OVERRIDE {
			PROFILE_ME;

			try {
				session->send_error(m_message_id, m_status_code, STD_MOVE(m_reason));
				session->shutdown_write();
			} catch(...){
				session->force_shutdown();
			}
		}
	};

	Session::Session(UniqueFile socket, boost::uint64_t max_request_length)
		: TcpSessionBase(STD_MOVE(socket))
		, m_max_request_length(max_request_length ? max_request_length : MainConfig::get<boost::uint64_t>("cbpp_max_request_length", 16384))
		, m_size_total(0), m_message_id(0), m_payload()
	{
	}
	Session::~Session(){
	}

	void Session::on_read_avail(StreamBuffer data){
		PROFILE_ME;

		try {
			m_size_total += data.size();
			if(m_size_total > m_max_request_length){
				DEBUG_THROW(Exception, ST_REQUEST_TOO_LARGE);
			}

			Reader::put_encoded_data(STD_MOVE(data));
		} catch(Exception &e){
			LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO,
				"Cbpp::Exception thrown: status_code = ", e.status_code(), ", what = ", e.what());
			enqueue_job(boost::make_shared<ErrorJob>(
				virtual_shared_from_this<Session>(), Reader::get_message_id(), e.status_code(), e.what()));
		}
	}
	void Session::on_data_message_header(boost::uint16_t message_id, boost::uint64_t /* payload_size */){
		PROFILE_ME;

		m_message_id = message_id;
		m_payload.clear();
	}
	void Session::on_data_message_payload(boost::uint64_t /* payload_offset */, StreamBuffer payload){
		PROFILE_ME;

		m_payload.splice(payload);
	}
	bool Session::on_data_message_end(boost::uint64_t /* payload_size */){
		PROFILE_ME;

		enqueue_job(boost::make_shared<DataMessageJob>(
			virtual_shared_from_this<Session>(), m_message_id, STD_MOVE(m_payload)));
		m_size_total = 0;

		return true;
	}

	bool Session::on_control_message(ControlCode control_code, boost::int64_t vint_param, std::string string_param){
		PROFILE_ME;

		enqueue_job(boost::make_shared<ControlMessageJob>(
			virtual_shared_from_this<Session>(), control_code, vint_param, STD_MOVE(string_param)));
		m_size_total = 0;
		m_message_id = 0;
		m_payload.clear();

		return true;
	}

	long Session::on_encoded_data_avail(StreamBuffer encoded){
		PROFILE_ME;

		return TcpSessionBase::send(STD_MOVE(encoded));
	}

	void Session::on_sync_control_message(ControlCode control_code, boost::int64_t vint_param, std::string string_param){
		PROFILE_ME;
		LOG_POSEIDON_DEBUG("Recevied control message from ", get_remote_info(),
			", control_code = ", control_code, ", vint_param = ", vint_param, ", string_param = ", string_param);

		switch(control_code){
		case CTL_PING:
			LOG_POSEIDON_TRACE("Received ping from ", get_remote_info());
			send(ControlMessage(ControlMessage::ID, ST_PONG, string_param));
			break;

		case CTL_SHUTDOWN:
			send(ControlMessage(ControlMessage::ID, ST_SHUTDOWN_REQUEST, string_param));
			shutdown_read();
			shutdown_write();
			break;

		case CTL_QUERY_MONO_CLOCK:
			send(ControlMessage(ControlMessage::ID, ST_MONOTONIC_CLOCK, boost::lexical_cast<std::string>(get_fast_mono_clock())));
			break;

		default:
			LOG_POSEIDON_WARNING("Unknown control code: ", control_code);
			DEBUG_THROW(Exception, ST_UNKNOWN_CTL_CODE, SharedNts(string_param));
		}
	}

	bool Session::send(boost::uint16_t message_id, StreamBuffer payload){
		PROFILE_ME;

		return Writer::put_data_message(message_id, STD_MOVE(payload));
	}
	bool Session::send_error(boost::uint16_t message_id, StatusCode status_code, std::string reason){
		PROFILE_ME;

		return Writer::put_control_message(message_id, status_code, STD_MOVE(reason));
	}
}

}
