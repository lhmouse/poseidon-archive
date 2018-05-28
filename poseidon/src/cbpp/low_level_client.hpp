// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_CBPP_LOW_LEVEL_CLIENT_HPP_
#define POSEIDON_CBPP_LOW_LEVEL_CLIENT_HPP_

#include "../tcp_client_base.hpp"
#include "reader.hpp"
#include "writer.hpp"
#include "status_codes.hpp"

namespace Poseidon {
namespace Cbpp {

class Low_level_client : public Tcp_client_base, protected Reader, protected Writer {
public:
	explicit Low_level_client(const Sock_addr &addr, bool use_ssl = false, bool verify_peer = true);
	~Low_level_client();

protected:
	// Tcp_client_base
	void on_connect() OVERRIDE;
	void on_read_hup() OVERRIDE;
	void on_close(int err_code) OVERRIDE;
	void on_receive(Stream_buffer data) OVERRIDE;

	// Reader
	void on_data_message_header(std::uint16_t message_id, std::uint64_t payload_size) OVERRIDE;
	void on_data_message_payload(std::uint64_t payload_offset, Stream_buffer payload) OVERRIDE;
	bool on_data_message_end(std::uint64_t payload_size) OVERRIDE;

	bool on_control_message(Status_code status_code, Stream_buffer param) OVERRIDE;

	// Writer
	long on_encoded_data_avail(Stream_buffer encoded) OVERRIDE;

	// 可覆写。
	virtual void on_low_level_data_message_header(std::uint16_t message_id, std::uint64_t payload_size) = 0;
	virtual void on_low_level_data_message_payload(std::uint64_t payload_offset, Stream_buffer payload) = 0;
	virtual bool on_low_level_data_message_end(std::uint64_t payload_size) = 0;

	virtual bool on_low_level_control_message(Status_code status_code, Stream_buffer param) = 0;

public:
	virtual bool send(std::uint16_t message_id, Stream_buffer payload);
	virtual bool send_control(Status_code status_code, Stream_buffer param);
	virtual bool shutdown(Status_code status_code, const char *reason = "") NOEXCEPT;
};

}
}

#endif
