// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2016, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "session.hpp"
#include "exception.hpp"
#include "control_message.hpp"
#include "../singletons/main_config.hpp"
#include "../singletons/job_dispatcher.hpp"
#include "../log.hpp"
#include "../profiler.hpp"
#include "../job_base.hpp"
#include "../time.hpp"

namespace Poseidon {

namespace Cbpp {
	class Session::SyncJobBase : public JobBase {
	private:
		const TcpSessionBase::DelayedShutdownGuard m_guard;
		const boost::weak_ptr<TcpSessionBase> m_category;
		const boost::weak_ptr<Session> m_weak_session;

	protected:
		explicit SyncJobBase(const boost::shared_ptr<Session> &session)
			: m_guard(session), m_category(session), m_weak_session(session)
		{
		}

	private:
		boost::weak_ptr<const void> get_category() const FINAL {
			return m_category;
		}
		void perform() FINAL {
			PROFILE_ME;

			const AUTO(session, m_weak_session.lock());
			if(!session || session->has_been_shutdown_write()){
				return;
			}

			try {
				really_perform(session);
			} catch(Exception &e){
				LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO,
					"Cbpp::Exception thrown: status_code = ", e.get_status_code(), ", what = ", e.what());
				try {
					session->send_error(ControlMessage::ID, e.get_status_code(), e.what());
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

	class Session::ReadHupJob : public Session::SyncJobBase {
	public:
		explicit ReadHupJob(const boost::shared_ptr<Session> &session)
			: SyncJobBase(session)
		{
		}

	protected:
		void really_perform(const boost::shared_ptr<Session> &session) OVERRIDE {
			PROFILE_ME;

			session->shutdown_write();
		}
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

	Session::Session(UniqueFile socket, boost::uint64_t max_request_length)
		: LowLevelSession(STD_MOVE(socket))
		, m_max_request_length(max_request_length ? max_request_length
		                                          : MainConfig::get<boost::uint64_t>("cbpp_max_request_length", 16384))
		, m_size_total(0), m_message_id(0), m_payload()
	{
	}
	Session::~Session(){
	}

	void Session::on_read_hup() NOEXCEPT
	try {
		PROFILE_ME;

		JobDispatcher::enqueue(
			boost::make_shared<ReadHupJob>(virtual_shared_from_this<Session>()),
			VAL_INIT);

		LowLevelSession::on_read_hup();
	} catch(std::exception &e){
		LOG_POSEIDON_WARNING("std::exception thrown: what = ", e.what());
		force_shutdown();
	} catch(...){
		LOG_POSEIDON_WARNING("Unknown exception thrown.");
		force_shutdown();
	}

	void Session::on_read_avail(StreamBuffer data){
		m_size_total += data.size();
		if(m_size_total > m_max_request_length){
			DEBUG_THROW(Exception, ST_REQUEST_TOO_LARGE);
		}

		LowLevelSession::on_read_avail(STD_MOVE(data));
	}

	void Session::on_low_level_data_message_header(boost::uint16_t message_id, boost::uint64_t payload_size){
		PROFILE_ME;

		(void)payload_size;

		m_size_total = 0;
		m_message_id = message_id;
		m_payload.clear();
	}
	void Session::on_low_level_data_message_payload(boost::uint64_t payload_offset, StreamBuffer payload){
		PROFILE_ME;

		(void)payload_offset;

		m_payload.splice(payload);
	}
	bool Session::on_low_level_data_message_end(boost::uint64_t payload_size){
		PROFILE_ME;

		(void)payload_size;

		JobDispatcher::enqueue(
			boost::make_shared<DataMessageJob>(virtual_shared_from_this<Session>(),
				m_message_id, STD_MOVE(m_payload)),
			VAL_INIT);

		return true;
	}

	bool Session::on_low_level_control_message(ControlCode control_code, boost::int64_t vint_param, std::string string_param){
		PROFILE_ME;

		JobDispatcher::enqueue(
			boost::make_shared<ControlMessageJob>(virtual_shared_from_this<Session>(),
				control_code, vint_param, STD_MOVE(string_param)),
			VAL_INIT);

		return true;
	}

	void Session::on_sync_control_message(ControlCode control_code, boost::int64_t vint_param, std::string string_param){
		PROFILE_ME;
		LOG_POSEIDON_DEBUG("Recevied control message from ", get_remote_info(),
			", control_code = ", control_code, ", vint_param = ", vint_param, ", string_param = ", string_param);

		switch(control_code){
		case CTL_PING:
			LOG_POSEIDON_TRACE("Received ping from ", get_remote_info());
			send(ControlMessage::ID,
				ControlMessage(ControlMessage::ID, ST_PONG, STD_MOVE(string_param)));
			break;

		case CTL_SHUTDOWN:
			send(ControlMessage::ID,
				ControlMessage(ControlMessage::ID, ST_SHUTDOWN_REQUEST, STD_MOVE(string_param)));
			shutdown_read();
			break;

		case CTL_QUERY_MONO_CLOCK:
			send(ControlMessage::ID,
				ControlMessage(ControlMessage::ID, ST_MONOTONIC_CLOCK, boost::lexical_cast<std::string>(get_fast_mono_clock())));
			break;

		default:
			LOG_POSEIDON_WARNING("Unknown control code: ", control_code);
			DEBUG_THROW(Exception, ST_UNKNOWN_CTL_CODE, sslit("Unknown control code"));
		}
	}
}

}
