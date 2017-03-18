// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2017, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_WEBSOCKET_LOW_LEVEL_CLIENT_HPP_
#define POSEIDON_WEBSOCKET_LOW_LEVEL_CLIENT_HPP_

#include "../http/upgraded_client_base.hpp"
#include "../mutex.hpp"
#include "opcodes.hpp"
#include "status_codes.hpp"
#include "reader.hpp"
#include "writer.hpp"

namespace Poseidon {

class TimerItem;

namespace WebSocket {
	class LowLevelClient : public Http::UpgradedClientBase, private Reader, private Writer {
	private:
		static void keep_alive_timer_proc(const boost::weak_ptr<LowLevelClient> &weak_client, boost::uint64_t now, boost::uint64_t period);

	private:
		mutable Mutex m_keep_alive_mutex;
		boost::shared_ptr<TimerItem> m_keep_alive_timer;
		boost::uint64_t m_last_pong_time;

	public:
		explicit LowLevelClient(const boost::shared_ptr<Http::LowLevelClient> &parent);
		~LowLevelClient();

	private:
		void create_keep_alive_timer(boost::uint64_t period);

	protected:
		// UpgradedSessionBase
		void on_receive(StreamBuffer data) OVERRIDE;

		// Reader
		void on_data_message_header(OpCode opcode) OVERRIDE;
		void on_data_message_payload(boost::uint64_t whole_offset, StreamBuffer payload) OVERRIDE;
		bool on_data_message_end(boost::uint64_t whole_size) OVERRIDE;

		bool on_control_message(OpCode opcode, StreamBuffer payload) OVERRIDE;

		// Writer
		long on_encoded_data_avail(StreamBuffer encoded) OVERRIDE;

		// 可覆写。
		virtual void on_low_level_message_header(OpCode opcode) = 0;
		virtual void on_low_level_message_payload(boost::uint64_t whole_offset, StreamBuffer payload) = 0;
		virtual bool on_low_level_message_end(boost::uint64_t whole_size) = 0;

		virtual bool on_low_level_control_message(OpCode opcode, StreamBuffer payload) = 0;

	public:
		bool shutdown_read() NOEXCEPT OVERRIDE {
			return shutdown(ST_NORMAL_CLOSURE);
		}
		bool shutdown_write() NOEXCEPT OVERRIDE {
			return shutdown(ST_NORMAL_CLOSURE);
		}

		bool send(OpCode opcode, StreamBuffer payload, bool masked = true);
		bool shutdown(StatusCode status_code, const char *reason = "") NOEXCEPT;
	};
}

}

#endif
