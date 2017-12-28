// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2017, LH_Mouse. All wrongs reserved.

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

class UpgradedSessionBase;
class HeaderOption;

class LowLevelClient : public TcpClientBase, protected ClientReader, protected ClientWriter {
	friend UpgradedSessionBase;

private:
	mutable Mutex m_upgraded_client_mutex;
	boost::shared_ptr<UpgradedSessionBase> m_upgraded_client;

public:
	explicit LowLevelClient(const SockAddr &addr, bool use_ssl = false, bool verify_peer = true);
	~LowLevelClient();

protected:
	const boost::shared_ptr<UpgradedSessionBase> &get_low_level_upgraded_client() const {
		// Epoll 线程读取不需要锁。
		return m_upgraded_client;
	}

	// TcpClientBase
	void on_connect() OVERRIDE;
	void on_read_hup() OVERRIDE;
	void on_close(int err_code) OVERRIDE;
	void on_receive(StreamBuffer data) OVERRIDE;

	// 注意，只能在 timer 线程中调用这些函数。
	void on_shutdown_timer(boost::uint64_t now) OVERRIDE;

	// ClientReader
	void on_response_headers(ResponseHeaders response_headers, boost::uint64_t content_length) OVERRIDE;
	void on_response_entity(boost::uint64_t entity_offset, StreamBuffer entity) OVERRIDE;
	bool on_response_end(boost::uint64_t content_length, OptionalMap headers) OVERRIDE;

	// ClientWriter
	long on_encoded_data_avail(StreamBuffer encoded) OVERRIDE;

	// 可覆写。
	virtual void on_low_level_response_headers(ResponseHeaders response_headers, boost::uint64_t content_length) = 0;
	virtual void on_low_level_response_entity(boost::uint64_t entity_offset, StreamBuffer entity) = 0;
	virtual boost::shared_ptr<UpgradedSessionBase> on_low_level_response_end(boost::uint64_t content_length, OptionalMap headers) = 0;

public:
	boost::shared_ptr<UpgradedSessionBase> get_upgraded_client() const;

	virtual bool send(RequestHeaders request_headers, StreamBuffer entity = StreamBuffer());
	virtual bool send(Verb verb, std::string uri, OptionalMap get_params = OptionalMap());
	virtual bool send(Verb verb, std::string uri, OptionalMap get_params, StreamBuffer entity, const HeaderOption &content_type);
	virtual bool send(Verb verb, std::string uri, OptionalMap get_params, OptionalMap headers, StreamBuffer entity = StreamBuffer());

	virtual bool send_chunked_header(RequestHeaders request_headers);
	virtual bool send_chunk(StreamBuffer entity);
	virtual bool send_chunked_trailer(OptionalMap headers);
};

}
}

#endif
