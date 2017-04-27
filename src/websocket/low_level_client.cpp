// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2017, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "low_level_client.hpp"
#include "exception.hpp"
#include "../singletons/timer_daemon.hpp"
#include "../singletons/main_config.hpp"
#include "../http/low_level_client.hpp"
#include "../log.hpp"
#include "../profiler.hpp"
#include "../time.hpp"
#include "../checked_arithmetic.hpp"
#include "../atomic.hpp"

namespace Poseidon {

namespace WebSocket {
	void LowLevelClient::keep_alive_timer_proc(const boost::weak_ptr<LowLevelClient> &weak_client, boost::uint64_t now, boost::uint64_t period){
		PROFILE_ME;

		const AUTO(client, weak_client.lock());
		if(!client){
			return;
		}

		const AUTO(interval_since_last_pong, saturated_sub(now, atomic_load(client->m_last_pong_time, ATOMIC_CONSUME)));
		if(interval_since_last_pong >= saturated_add(period, period)){
			LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO,
				"No pong received since the last two keep alive intervals. Shut down the connection.");
			client->force_shutdown();
			return;
		}

		const AUTO(utc_now, get_utc_time());
		char str[64];
		unsigned len = (unsigned)std::sprintf(str, "%llu", (unsigned long long)utc_now);
		client->send(OP_PING, StreamBuffer(str, len));
	}

	LowLevelClient::LowLevelClient(const boost::shared_ptr<Http::LowLevelClient> &parent)
		: Http::UpgradedSessionBase(parent), Reader(false), Writer()
		, m_last_pong_time((boost::uint64_t)-1)
	{
	}
	LowLevelClient::~LowLevelClient(){
	}

	void LowLevelClient::create_keep_alive_timer(){
		PROFILE_ME;

		const Mutex::UniqueLock lock(m_keep_alive_mutex);
		if(m_keep_alive_timer){
			return;
		}
		const AUTO(keep_alive_timeout, MainConfig::get<boost::uint64_t>("websocket_keep_alive_timeout", 30000));
		m_keep_alive_timer = TimerDaemon::register_low_level_timer(0, keep_alive_timeout / 2,
			boost::bind(&keep_alive_timer_proc, virtual_weak_from_this<LowLevelClient>(), _2, _3));
	}

	void LowLevelClient::on_connect(){
	}
	void LowLevelClient::on_read_hup(){
	}
	void LowLevelClient::on_close(int err_code){
		(void)err_code;
	}
	void LowLevelClient::on_receive(StreamBuffer data){
		PROFILE_ME;

		Reader::put_encoded_data(STD_MOVE(data));
	}

	void LowLevelClient::on_data_message_header(OpCode opcode){
		PROFILE_ME;

		on_low_level_message_header(opcode);
	}
	void LowLevelClient::on_data_message_payload(boost::uint64_t whole_offset, StreamBuffer payload){
		PROFILE_ME;

		on_low_level_message_payload(whole_offset, STD_MOVE(payload));
	}
	bool LowLevelClient::on_data_message_end(boost::uint64_t whole_size){
		PROFILE_ME;

		const AUTO(now, get_fast_mono_clock());
		atomic_store(m_last_pong_time, now, ATOMIC_RELEASE);
		create_keep_alive_timer();

		return on_low_level_message_end(whole_size);
	}

	bool LowLevelClient::on_control_message(OpCode opcode, StreamBuffer payload){
		PROFILE_ME;

		const AUTO(now, get_fast_mono_clock());
		atomic_store(m_last_pong_time, now, ATOMIC_RELEASE);
		create_keep_alive_timer();

		return on_low_level_control_message(opcode, STD_MOVE(payload));
	}

	long LowLevelClient::on_encoded_data_avail(StreamBuffer encoded){
		PROFILE_ME;

		return UpgradedSessionBase::send(STD_MOVE(encoded));
	}

	bool LowLevelClient::send(OpCode opcode, StreamBuffer payload, bool masked){
		PROFILE_ME;

		return Writer::put_message(opcode, masked, STD_MOVE(payload));
	}

	bool LowLevelClient::shutdown(StatusCode status_code, const char *reason) NOEXCEPT {
		PROFILE_ME;

		const AUTO(parent, get_parent());
		if(!parent){
			return false;
		}

		try {
			Writer::put_close_message(status_code, true, StreamBuffer(reason));
			parent->shutdown_read();
			return parent->shutdown_write();
		} catch(std::exception &e){
			LOG_POSEIDON_ERROR("std::exception thrown: what = ", e.what());
			parent->force_shutdown();
			return false;
		} catch(...){
			LOG_POSEIDON_ERROR("Unknown exception thrown.");
			parent->force_shutdown();
			return false;
		}
	}
}

}
