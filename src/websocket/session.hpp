// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_WEBSOCKET_SESSION_HPP_
#define POSEIDON_WEBSOCKET_SESSION_HPP_

#include "low_level_session.hpp"

namespace Poseidon {
namespace WebSocket {

class Session : public LowLevelSession {
private:
	class SyncJobBase;
	class ReadHupJob;
	class PingJob;
	class DataMessageJob;
	class ControlMessageJob;

private:
	volatile boost::uint64_t m_max_request_length;
	boost::uint64_t m_size_total;
	OpCode m_opcode;
	StreamBuffer m_payload;

public:
	explicit Session(const boost::shared_ptr<Http::LowLevelSession> &parent);
	~Session();

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
	void on_read_hup() OVERRIDE;
	void on_shutdown_timer(boost::uint64_t now) OVERRIDE;

	// LowLevelSession
	void on_low_level_message_header(OpCode opcode) OVERRIDE;
	void on_low_level_message_payload(boost::uint64_t whole_offset, StreamBuffer payload) OVERRIDE;
	bool on_low_level_message_end(boost::uint64_t whole_size) OVERRIDE;

	bool on_low_level_control_message(OpCode opcode, StreamBuffer payload) OVERRIDE;

	// 可覆写。
	virtual void on_sync_data_message(OpCode opcode, StreamBuffer payload) = 0;
	virtual void on_sync_control_message(OpCode opcode, StreamBuffer payload);

public:
	boost::uint64_t get_max_request_length() const;
	void set_max_request_length(boost::uint64_t max_request_length);
};

}
}

#endif
