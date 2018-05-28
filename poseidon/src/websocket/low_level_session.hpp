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
namespace Websocket {

class Low_level_session : public Http::Upgraded_session_base, protected Reader, protected Writer {
public:
	explicit Low_level_session(const boost::shared_ptr<Http::Low_level_session> &parent);
	~Low_level_session();

protected:
	// Upgraded_session_base
	void on_connect() OVERRIDE;
	void on_read_hup() OVERRIDE;
	void on_close(int err_code) OVERRIDE;
	void on_receive(Stream_buffer data) OVERRIDE;

	// Reader
	void on_data_message_header(Op_code opcode) OVERRIDE;
	void on_data_message_payload(std::uint64_t whole_offset, Stream_buffer payload) OVERRIDE;
	bool on_data_message_end(std::uint64_t whole_size) OVERRIDE;

	bool on_control_message(Op_code opcode, Stream_buffer payload) OVERRIDE;

	// Writer
	long on_encoded_data_avail(Stream_buffer encoded) OVERRIDE;

	// 可覆写。
	virtual void on_low_level_message_header(Op_code opcode) = 0;
	virtual void on_low_level_message_payload(std::uint64_t whole_offset, Stream_buffer payload) = 0;
	virtual bool on_low_level_message_end(std::uint64_t whole_size) = 0;

	virtual bool on_low_level_control_message(Op_code opcode, Stream_buffer payload) = 0;

public:
	virtual bool send(Op_code opcode, Stream_buffer payload, bool masked = false);
	virtual bool shutdown(Status_code status_code, const char *reason = "") NOEXCEPT;
};

}
}

#endif
