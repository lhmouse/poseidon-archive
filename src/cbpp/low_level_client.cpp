// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2017, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "low_level_client.hpp"
#include "exception.hpp"
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

		const AUTO(utc_now, get_utc_time());
		char str[64];
		unsigned len = (unsigned)std::sprintf(str, "%llu", (unsigned long long)utc_now);
		client->send_control(Poseidon::Cbpp::ST_PING, StreamBuffer(str, len));
	}

	LowLevelClient::LowLevelClient(const SockAddr &addr, bool use_ssl, bool verify_peer, boost::uint64_t keep_alive_interval)
		: TcpClientBase(addr, use_ssl, verify_peer), Reader(), Writer()
		, m_keep_alive_interval(keep_alive_interval)
		, m_last_pong_time((boost::uint64_t)-1)
	{
	}
	LowLevelClient::LowLevelClient(const IpPort &addr, bool use_ssl, bool verify_peer, boost::uint64_t keep_alive_interval)
		: TcpClientBase(addr, use_ssl, verify_peer)
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

	bool LowLevelClient::on_control_message(StatusCode status_code, StreamBuffer param){
		PROFILE_ME;

		m_last_pong_time = get_fast_mono_clock();

		return on_low_level_control_message(status_code, STD_MOVE(param));
	}

	long LowLevelClient::on_encoded_data_avail(StreamBuffer encoded){
		PROFILE_ME;

		if(!m_keep_alive_timer && (m_keep_alive_interval != 0)){
			m_keep_alive_timer = TimerDaemon::register_timer(m_keep_alive_interval, m_keep_alive_interval,
				boost::bind(&keep_alive_timer_proc, virtual_weak_from_this<LowLevelClient>(), _2, _3));
		}

		return TcpClientBase::send(STD_MOVE(encoded));
	}

	bool LowLevelClient::send(boost::uint16_t message_id, StreamBuffer payload){
		PROFILE_ME;

		return Writer::put_data_message(message_id, STD_MOVE(payload));
	}
	bool LowLevelClient::send_control(StatusCode status_code, StreamBuffer param){
		PROFILE_ME;

		return Writer::put_control_message(status_code, STD_MOVE(param));
	}
	bool LowLevelClient::shutdown(StatusCode status_code, const char *reason) NOEXCEPT {
		PROFILE_ME;

		try {
			Writer::put_control_message(status_code, StreamBuffer(reason));
			shutdown_read();
			return shutdown_write();
		} catch(std::exception &e){
			LOG_POSEIDON_ERROR("std::exception thrown: what = ", e.what());
			force_shutdown();
			return false;
		} catch(...){
			LOG_POSEIDON_ERROR("Unknown exception thrown.");
			force_shutdown();
			return false;
		}
	}
}

}
