// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_CBPP_SESSION_HPP_
#define POSEIDON_CBPP_SESSION_HPP_

#include "low_level_session.hpp"

namespace Poseidon {
namespace Cbpp {

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
	unsigned m_message_id;
	Stream_buffer m_payload;

public:
	explicit Session(Move<Unique_file> socket);
	~Session();

protected:
	boost::uint64_t get_low_level_size_total() const {
		return m_size_total;
	}
	unsigned get_low_level_message_id() const {
		return m_message_id;
	}
	const Stream_buffer &get_low_level_payload() const {
		return m_payload;
	}

	// Tcp_session_base
	void on_read_hup() OVERRIDE;
	void on_shutdown_timer(boost::uint64_t now) OVERRIDE;

	// Low_level_session
	void on_low_level_data_message_header(boost::uint16_t message_id, boost::uint64_t payload_size) OVERRIDE;
	void on_low_level_data_message_payload(boost::uint64_t payload_offset, Stream_buffer payload) OVERRIDE;
	bool on_low_level_data_message_end(boost::uint64_t payload_size) OVERRIDE;

	bool on_low_level_control_message(Status_code status_code, Stream_buffer param) OVERRIDE;

	// 可覆写。
	virtual void on_sync_data_message(boost::uint16_t message_id, Stream_buffer payload) = 0;
	virtual void on_sync_control_message(Status_code status_code, Stream_buffer param);

public:
	boost::uint64_t get_max_request_length() const;
	void set_max_request_length(boost::uint64_t max_request_length);
};

}
}

#endif
