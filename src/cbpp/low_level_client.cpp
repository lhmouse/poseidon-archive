// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2016, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "low_level_client.hpp"
#include "exception.hpp"
#include "control_message.hpp"
#include "../singletons/timer_daemon.hpp"
#include "../log.hpp"
#include "../profiler.hpp"
#include "../time.hpp"

namespace Poseidon {

namespace Cbpp {
	void LowLevelClient::keep_alive_timer_proc(const boost::weak_ptr<LowLevelClient> &weak_client, boost::uint64_t now, boost::uint64_t period){
		PROFILE_ME;

		const AUTO(client, weak_client.lock());
		if(!client){
			return;
		}

		if(client->m_last_pong_time < now - period * 2){
			LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO,
				"No pong received since the last two keep alive intervals. Shut down the connection.");
			client->force_shutdown();
			return;
		}

		client->send_control(CTL_PING, 0, boost::lexical_cast<std::string>(get_utc_time()));
	}

	LowLevelClient::LowLevelClient(const SockAddr &addr, bool use_ssl, boost::uint64_t keep_alive_interval)
		: TcpClientBase(addr, use_ssl)
		, m_keep_alive_interval(keep_alive_interval)
		, m_last_pong_time((boost::uint64_t)-1)
	{
	}
	LowLevelClient::LowLevelClient(const IpPort &addr, bool use_ssl, boost::uint64_t keep_alive_interval)
		: TcpClientBase(addr, use_ssl)
		, m_keep_alive_interval(keep_alive_interval)
		, m_last_pong_time((boost::uint64_t)-1)
	{
	}
	LowLevelClient::~LowLevelClient(){
	}

	void LowLevelClient::on_read_avail(StreamBuffer data){
		PROFILE_ME;

		Reader::put_encoded_data(STD_MOVE(data));
	}

	void LowLevelClient::on_data_message_header(boost::uint16_t message_id, boost::uint64_t payload_size){
		PROFILE_ME;

		on_low_level_data_message_header(message_id, payload_size);
	}
	void LowLevelClient::on_data_message_payload(boost::uint64_t payload_offset, StreamBuffer payload){
		PROFILE_ME;

		on_low_level_data_message_payload(payload_offset, STD_MOVE(payload));
	}
	bool LowLevelClient::on_data_message_end(boost::uint64_t payload_size){
		PROFILE_ME;

		m_last_pong_time = get_fast_mono_clock();

		return on_low_level_data_message_end(payload_size);
	}

	bool LowLevelClient::on_control_message(ControlCode control_code, boost::int64_t vint_param, std::string string_param){
		PROFILE_ME;

		m_last_pong_time = get_fast_mono_clock();

		return on_low_level_error_message(control_code, vint_param, STD_MOVE(string_param));
	}

	long LowLevelClient::on_encoded_data_avail(StreamBuffer encoded){
		PROFILE_ME;

		if(!m_keep_alive_timer){
			m_keep_alive_timer = TimerDaemon::register_timer(m_keep_alive_interval, m_keep_alive_interval,
				boost::bind(&keep_alive_timer_proc, virtual_weak_from_this<LowLevelClient>(), _2, _3));
		}

		return TcpClientBase::send(STD_MOVE(encoded));
	}

	bool LowLevelClient::send(boost::uint16_t message_id, StreamBuffer payload){
		PROFILE_ME;

		return Writer::put_data_message(message_id, STD_MOVE(payload));
	}
	bool LowLevelClient::send_control(ControlCode control_code, boost::int64_t vint_param, std::string string_param){
		PROFILE_ME;

		return Writer::put_control_message(control_code, vint_param, STD_MOVE(string_param));
	}
}

}
