// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_CBPP_CLIENT_HPP_
#define POSEIDON_CBPP_CLIENT_HPP_

#include "../tcp_client_base.hpp"
#include "reader.hpp"
#include "writer.hpp"
#include "control_codes.hpp"
#include "status_codes.hpp"

namespace Poseidon {

class TimerItem;

namespace Cbpp {
	class Client : public TcpClientBase, private Reader, private Writer {
	private:
		class SyncJobBase;
		class ConnectJob;
		class DataMessageHeaderJob;
		class DataMessagePayloadJob;
		class DataMessageEndJob;
		class ErrorMessageJob;

	private:
		static void keep_alive_timer_proc(const boost::weak_ptr<Client> &weak_client, boost::uint64_t now, boost::uint64_t period);

	private:
		const boost::uint64_t m_keep_alive_interval;

		boost::shared_ptr<TimerItem> m_keep_alive_timer;
		boost::uint64_t m_last_pong_time;

	protected:
		Client(const SockAddr &addr, bool use_ssl, boost::uint64_t keep_alive_interval);
		Client(const IpPort &addr, bool use_ssl, boost::uint64_t keep_alive_interval);
		~Client();

	protected:
		// TcpSessionBase
		void on_connect() OVERRIDE;

		void on_read_avail(StreamBuffer data) OVERRIDE;

		// Reader
		void on_data_message_header(boost::uint16_t message_id, boost::uint64_t payload_size) OVERRIDE;
		void on_data_message_payload(boost::uint64_t payload_offset, StreamBuffer payload) OVERRIDE;
		bool on_data_message_end(boost::uint64_t payload_size) OVERRIDE;

		bool on_control_message(ControlCode control_code, boost::int64_t vint_param, std::string string_param) OVERRIDE;

		// Writer
		long on_encoded_data_avail(StreamBuffer encoded) OVERRIDE;

		// 可覆写。
		virtual void on_sync_connect();

		virtual void on_sync_data_message_header(boost::uint16_t message_id, boost::uint64_t payload_size) = 0;
		virtual void on_sync_data_message_payload(boost::uint64_t payload_offset, StreamBuffer payload) = 0;
		virtual void on_sync_data_message_end(boost::uint64_t payload_size) = 0;

		virtual void on_sync_error_message(boost::uint16_t message_id, StatusCode status_code, std::string reason);

	public:
		bool send(boost::uint16_t message_id, StreamBuffer payload);
		bool send_control(ControlCode control_code, boost::int64_t vint_param, std::string string_param);

		template<typename MsgT>
		bool send(const MsgT &msg){
			return send(MsgT::ID, StreamBuffer(msg));
		}
	};
}

}

#endif
