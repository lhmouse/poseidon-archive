// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2016, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "low_level_client.hpp"
#include "exception.hpp"
#include "../singletons/timer_daemon.hpp"
#include "../http/low_level_client.hpp"
#include "../log.hpp"
#include "../profiler.hpp"
#include "../time.hpp"

namespace Poseidon {

namespace WebSocket {
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
		client->send(OP_PING, StreamBuffer(str, len));
	}

	LowLevelClient::LowLevelClient(const boost::shared_ptr<Http::LowLevelClient> &parent, boost::uint64_t keep_alive_interval)
		: Http::UpgradedClientBase(parent), Reader(false), Writer()
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

		return on_low_level_message_end(whole_size);
	}

	bool LowLevelClient::on_control_message(OpCode opcode, StreamBuffer payload){
		PROFILE_ME;

		return on_low_level_control_message(opcode, STD_MOVE(payload));
	}

	long LowLevelClient::on_encoded_data_avail(StreamBuffer encoded){
		PROFILE_ME;

		if(!m_keep_alive_timer){
			m_keep_alive_timer = TimerDaemon::register_timer(m_keep_alive_interval, m_keep_alive_interval,
				boost::bind(&keep_alive_timer_proc, virtual_weak_from_this<LowLevelClient>(), _2, _3));
		}

		return UpgradedClientBase::send(STD_MOVE(encoded));
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
