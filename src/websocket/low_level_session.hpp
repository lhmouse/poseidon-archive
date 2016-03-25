// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2016, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_WEBSOCKET_LOW_LEVEL_SESSION_HPP_
#define POSEIDON_WEBSOCKET_LOW_LEVEL_SESSION_HPP_

#include "../http/upgraded_session_base.hpp"
#include "opcodes.hpp"
#include "status_codes.hpp"
#include "reader.hpp"
#include "writer.hpp"

namespace Poseidon {

namespace WebSocket {
	class LowLevelSession : public Http::UpgradedSessionBase, private Reader, private Writer {
	private:
		const boost::uint64_t m_max_request_length;

		boost::uint64_t m_size_total;
		OpCode m_opcode;
		StreamBuffer m_payload;

	public:
		explicit LowLevelSession(const boost::shared_ptr<Http::LowLevelSession> &parent, boost::uint64_t max_request_length = 0);
		~LowLevelSession();

	protected:
		boost::uint64_t get_low_level_size_total() const {
			return m_size_total;
		}
		OpCode get_low_level_opcode() const {
			return m_opcode;
		}
		const StreamBuffer &get_low_level_payload() const {
			return m_payload;
		}

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
		virtual bool on_low_level_data_message(OpCode opcode, StreamBuffer payload) = 0;
		virtual bool on_low_level_control_message(OpCode opcode, StreamBuffer payload) = 0;

	public:
		bool shutdown_read() NOEXCEPT OVERRIDE {
			return shutdown(ST_NORMAL_CLOSURE);
		}
		bool shutdown_write() NOEXCEPT OVERRIDE {
			return shutdown(ST_NORMAL_CLOSURE);
		}

		bool send(OpCode opcode, StreamBuffer payload, bool masked = false);
		bool shutdown(StatusCode status_code, StreamBuffer additional = StreamBuffer()) NOEXCEPT;
	};
}

}

#endif
