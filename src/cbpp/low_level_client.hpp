// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2016, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_CBPP_LOW_LEVEL_CLIENT_HPP_
#define POSEIDON_CBPP_LOW_LEVEL_CLIENT_HPP_

#include "../tcp_client_base.hpp"
#include "reader.hpp"
#include "writer.hpp"
#include "status_codes.hpp"

namespace Poseidon {

class TimerItem;

namespace Cbpp {
	class LowLevelClient : public TcpClientBase, private Reader, private Writer {
	private:
		static void keep_alive_timer_proc(const boost::weak_ptr<LowLevelClient> &weak_client, boost::uint64_t now, boost::uint64_t period);

	private:
		const boost::uint64_t m_keep_alive_interval;

		boost::shared_ptr<TimerItem> m_keep_alive_timer;
		boost::uint64_t m_last_pong_time;

	protected:
		LowLevelClient(const SockAddr &addr, bool use_ssl, boost::uint64_t keep_alive_interval);
		LowLevelClient(const IpPort &addr, bool use_ssl, boost::uint64_t keep_alive_interval);
		~LowLevelClient();

	protected:
		// TcpSessionBase
		void on_read_avail(StreamBuffer data) OVERRIDE;

		// Reader
		void on_data_message_header(boost::uint16_t message_id, boost::uint64_t payload_size) OVERRIDE;
		void on_data_message_payload(boost::uint64_t payload_offset, StreamBuffer payload) OVERRIDE;
		bool on_data_message_end(boost::uint64_t payload_size) OVERRIDE;

		bool on_control_message(StatusCode status_code, StreamBuffer param) OVERRIDE;

		// Writer
		long on_encoded_data_avail(StreamBuffer encoded) OVERRIDE;

		// 可覆写。
		virtual void on_low_level_data_message_header(boost::uint16_t message_id, boost::uint64_t payload_size) = 0;
		virtual void on_low_level_data_message_payload(boost::uint64_t payload_offset, StreamBuffer payload) = 0;
		virtual bool on_low_level_data_message_end(boost::uint64_t payload_size) = 0;

		virtual bool on_low_level_control_message(StatusCode status_code, StreamBuffer param) = 0;

	public:
		bool send(boost::uint16_t message_id, StreamBuffer payload);
		bool send_control(StatusCode status_code, StreamBuffer param);
		bool shutdown(StatusCode status_code, const char *reason = "") NOEXCEPT;
	};
}

}

#endif
