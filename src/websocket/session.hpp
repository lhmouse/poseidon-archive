// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_WEBSOCKET_SESSION_HPP_
#define POSEIDON_WEBSOCKET_SESSION_HPP_

#include "../http/upgraded_session_base.hpp"
#include "opcodes.hpp"
#include "status_codes.hpp"
#include "reader.hpp"
#include "writer.hpp"

namespace Poseidon {

namespace WebSocket {
	class Session : public Http::UpgradedSessionBase, private Reader, private Writer {
	private:
		class SyncJobBase;
		class DataMessageJob;
		class ControlMessageJob;
		class ErrorJob;

	private:
		const boost::uint64_t m_max_request_length;

		boost::uint64_t m_size_total;
		OpCode m_opcode;
		StreamBuffer m_payload;

	public:
		explicit Session(const boost::shared_ptr<Http::Session> &parent, boost::uint64_t max_request_length = 0);
		~Session();

	protected:
		// UpgradedSessionBase
		void on_read_avail(StreamBuffer data) OVERRIDE;

		// Reader
		void on_data_message_header(OpCode opcode) OVERRIDE;
		void on_data_message_payload(boost::uint64_t whole_offset, StreamBuffer payload) OVERRIDE;
		bool on_data_message_end(boost::uint64_t whole_size) OVERRIDE;

		bool on_control_message(OpCode opcode, StreamBuffer payload) OVERRIDE;

		// Writer
		long on_encoded_data_avail(StreamBuffer encoded) OVERRIDE;

		// 可覆写。
		virtual void on_sync_data_message(OpCode opcode, StreamBuffer payload) = 0;

		virtual void on_sync_control_message(OpCode opcode, StreamBuffer payload);

	public:
		bool shutdown_read() NOEXCEPT OVERRIDE {
			return shutdown(ST_NORMAL_CLOSURE);
		}
		bool shutdown_write() NOEXCEPT OVERRIDE {
			return shutdown(ST_NORMAL_CLOSURE);
		}

		bool send(StreamBuffer payload, bool binary, bool masked = false);

		bool shutdown(StatusCode status_code, StreamBuffer additional = StreamBuffer()) NOEXCEPT;
	};
}

}

#endif
