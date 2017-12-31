// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_WEBSOCKET_LOW_LEVEL_SESSION_HPP_
#define POSEIDON_WEBSOCKET_LOW_LEVEL_SESSION_HPP_

#include "../http/upgraded_session_base.hpp"
#include "opcodes.hpp"
#include "status_codes.hpp"
#include "reader.hpp"
#include "writer.hpp"

namespace Poseidon {
namespace WebSocket {

class LowLevelSession : public Http::UpgradedSessionBase, protected Reader, protected Writer {
public:
	explicit LowLevelSession(const boost::shared_ptr<Http::LowLevelSession> &parent);
	~LowLevelSession();

protected:
	// UpgradedSessionBase
	void on_connect() OVERRIDE;
	void on_read_hup() OVERRIDE;
	void on_close(int err_code) OVERRIDE;
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
	virtual bool send(OpCode opcode, StreamBuffer payload, bool masked = false);
	virtual bool shutdown(StatusCode status_code, const char *reason = "") NOEXCEPT;
};

}
}

#endif
