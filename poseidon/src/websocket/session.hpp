// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_WEBSOCKET_SESSION_HPP_
#define POSEIDON_WEBSOCKET_SESSION_HPP_

#include "low_level_session.hpp"

namespace Poseidon {
namespace Websocket {

class Session : public Low_level_session {
private:
	class Sync_job_base;
	class Read_hup_job;
	class Ping_job;
	class Data_message_job;
	class Control_message_job;

private:
	volatile boost::uint64_t m_max_request_length;
	boost::uint64_t m_size_total;
	Opcode m_opcode;
	Stream_buffer m_payload;

public:
	explicit Session(const boost::shared_ptr<Http::Low_level_session> &parent);
	~Session();

protected:
	boost::uint64_t get_low_level_size_total() const {
		return m_size_total;
	}
	Opcode get_low_level_opcode() const {
		return m_opcode;
	}
	const Stream_buffer &get_low_level_payload() const {
		return m_payload;
	}

	// Upgraded_session_base
	void on_read_hup() OVERRIDE;
	void on_shutdown_timer(boost::uint64_t now) OVERRIDE;

	// Low_level_session
	void on_low_level_message_header(Opcode opcode) OVERRIDE;
	void on_low_level_message_payload(boost::uint64_t whole_offset, Stream_buffer payload) OVERRIDE;
	bool on_low_level_message_end(boost::uint64_t whole_size) OVERRIDE;

	bool on_low_level_control_message(Opcode opcode, Stream_buffer payload) OVERRIDE;

	// 可覆写。
	virtual void on_sync_data_message(Opcode opcode, Stream_buffer payload) = 0;
	virtual void on_sync_control_message(Opcode opcode, Stream_buffer payload);

public:
	boost::uint64_t get_max_request_length() const;
	void set_max_request_length(boost::uint64_t max_request_length);
};

}
}

#endif
