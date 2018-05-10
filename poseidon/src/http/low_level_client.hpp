// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_HTTP_LOW_LEVEL_CLIENT_HPP_
#define POSEIDON_HTTP_LOW_LEVEL_CLIENT_HPP_

#include "../tcp_client_base.hpp"
#include "../mutex.hpp"
#include "client_reader.hpp"
#include "client_writer.hpp"
#include "request_headers.hpp"
#include "response_headers.hpp"
#include "status_codes.hpp"

namespace Poseidon {
namespace Http {

class Upgraded_session_base;
class Header_option;

class Low_level_client : public Tcp_client_base, protected Client_reader, protected Client_writer {
	friend Upgraded_session_base;

private:
	mutable Mutex m_upgraded_client_mutex;
	boost::shared_ptr<Upgraded_session_base> m_upgraded_client;

public:
	explicit Low_level_client(const Sock_addr &addr, bool use_ssl = false, bool verify_peer = true);
	~Low_level_client();

protected:
	const boost::shared_ptr<Upgraded_session_base> &get_low_level_upgraded_client() const {
		// Epoll 线程读取不需要锁。
		return m_upgraded_client;
	}

	// Tcp_client_base
	void on_connect() OVERRIDE;
	void on_read_hup() OVERRIDE;
	void on_close(int err_code) OVERRIDE;
	void on_receive(Stream_buffer data) OVERRIDE;

	// 注意，只能在 timer 线程中调用这些函数。
	void on_shutdown_timer(boost::uint64_t now) OVERRIDE;

	// Client_reader
	void on_response_headers(Response_headers response_headers, boost::uint64_t content_length) OVERRIDE;
	void on_response_entity(boost::uint64_t entity_offset, Stream_buffer entity) OVERRIDE;
	bool on_response_end(boost::uint64_t content_length, Optional_map headers) OVERRIDE;

	// Client_writer
	long on_encoded_data_avail(Stream_buffer encoded) OVERRIDE;

	// 可覆写。
	virtual void on_low_level_response_headers(Response_headers response_headers, boost::uint64_t content_length) = 0;
	virtual void on_low_level_response_entity(boost::uint64_t entity_offset, Stream_buffer entity) = 0;
	virtual boost::shared_ptr<Upgraded_session_base> on_low_level_response_end(boost::uint64_t content_length, Optional_map headers) = 0;

public:
	boost::shared_ptr<Upgraded_session_base> get_upgraded_client() const;

	virtual bool send(Request_headers request_headers, Stream_buffer entity = Stream_buffer());
	virtual bool send(Verb verb, std::string uri, Optional_map get_params = Optional_map());
	virtual bool send(Verb verb, std::string uri, Optional_map get_params, Stream_buffer entity, const Header_option &content_type);
	virtual bool send(Verb verb, std::string uri, Optional_map get_params, Optional_map headers, Stream_buffer entity = Stream_buffer());

	virtual bool send_chunked_header(Request_headers request_headers);
	virtual bool send_chunk(Stream_buffer entity);
	virtual bool send_chunked_trailer(Optional_map headers);
};

}
}

#endif
