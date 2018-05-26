// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_WEBSOCKET_CLIENT_HPP_
#define POSEIDON_WEBSOCKET_CLIENT_HPP_

#include "low_level_client.hpp"

namespace Poseidon {
namespace Websocket {

class Client : public Low_level_client {
private:
	class Sync_job_base;
	class Connect_job;
	class Read_hup_job;
	class Data_message_job;
	class Control_message_job;

private:
	Op_code m_opcode;
	Stream_buffer m_payload;

public:
	explicit Client(const boost::shared_ptr<Http::Low_level_client> &parent);
	~Client();

protected:
	Op_code get_low_level_opcode() const {
		return m_opcode;
	}
	const Stream_buffer &get_low_level_payload() const {
		return m_payload;
	}

	// Upgraded_session_base
	void on_connect() OVERRIDE;
	void on_read_hup() OVERRIDE;

	// Low_level_client
	void on_low_level_message_header(Op_code opcode) OVERRIDE;
	void on_low_level_message_payload(boost::uint64_t whole_offset, Stream_buffer payload) OVERRIDE;
	bool on_low_level_message_end(boost::uint64_t whole_size) OVERRIDE;

	bool on_low_level_control_message(Op_code opcode, Stream_buffer payload) OVERRIDE;

	// 可覆写。
	virtual void on_sync_connect();

	virtual void on_sync_data_message(Op_code opcode, Stream_buffer payload) = 0;
	virtual void on_sync_control_message(Op_code opcode, Stream_buffer payload);
};

}
}

#endif
