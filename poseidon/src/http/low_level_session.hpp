// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_HTTP_LOW_LEVEL_SESSION_HPP_
#define POSEIDON_HTTP_LOW_LEVEL_SESSION_HPP_

#include "../tcp_session_base.hpp"
#include "../mutex.hpp"
#include "server_reader.hpp"
#include "server_writer.hpp"
#include "request_headers.hpp"
#include "response_headers.hpp"
#include "status_codes.hpp"

namespace Poseidon {
namespace Http {

class Upgraded_session_base;
class Header_option;

class Low_level_session : public Tcp_session_base, protected Server_reader, protected Server_writer {
	friend Upgraded_session_base;

private:
	mutable Mutex m_upgraded_session_mutex;
	boost::shared_ptr<Upgraded_session_base> m_upgraded_session;

public:
	explicit Low_level_session(Move<Unique_file> socket);
	~Low_level_session();

protected:
	const boost::shared_ptr<Upgraded_session_base> &get_low_level_upgraded_session() const {
		// Epoll 线程读取不需要锁。
		return m_upgraded_session;
	}

	// Tcp_session_base
	void on_connect() OVERRIDE;
	void on_read_hup() OVERRIDE;
	void on_close(int err_code) OVERRIDE;
	void on_receive(Stream_buffer data) OVERRIDE;

	// 注意，只能在 timer 线程中调用这些函数。
	void on_shutdown_timer(boost::uint64_t now) OVERRIDE;

	// Server_reader
	void on_request_headers(Request_headers request_headers, boost::uint64_t content_length) OVERRIDE;
	void on_request_entity(boost::uint64_t entity_offset, Stream_buffer entity) OVERRIDE;
	bool on_request_end(boost::uint64_t content_length, Option_map headers) OVERRIDE;

	// Server_writer
	long on_encoded_data_avail(Stream_buffer encoded) OVERRIDE;

	// 可覆写。
	virtual void on_low_level_request_headers(Request_headers request_headers, boost::uint64_t content_length) = 0;
	virtual void on_low_level_request_entity(boost::uint64_t entity_offset, Stream_buffer entity) = 0;
	virtual boost::shared_ptr<Upgraded_session_base> on_low_level_request_end(boost::uint64_t content_length, Option_map headers) = 0;

public:
	boost::shared_ptr<Upgraded_session_base> get_upgraded_session() const;

	virtual bool send(Response_headers response_headers, Stream_buffer entity = Stream_buffer());
	virtual bool send(Status_code status_code);
	virtual bool send(Status_code status_code, Stream_buffer entity, const Header_option &content_type);
	virtual bool send(Status_code status_code, Option_map headers, Stream_buffer entity = Stream_buffer());

	virtual bool send_chunked_header(Response_headers response_headers);
	virtual bool send_chunk(Stream_buffer entity);
	virtual bool send_chunked_trailer(Option_map headers = Option_map());

	virtual bool send_default(Status_code status_code, Option_map headers = Option_map());
	virtual bool send_default_and_shutdown(Status_code status_code, const Option_map &headers = Option_map()) NOEXCEPT;
	virtual bool send_default_and_shutdown(Status_code status_code, Move<Option_map> headers) NOEXCEPT;
};

}
}

#endif
