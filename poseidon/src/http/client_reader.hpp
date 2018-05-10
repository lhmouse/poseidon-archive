// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_HTTP_CLIENT_READER_HPP_
#define POSEIDON_HTTP_CLIENT_READER_HPP_

#include <string>
#include <cstddef>
#include <boost/cstdint.hpp>
#include "../stream_buffer.hpp"
#include "../optional_map.hpp"
#include "response_headers.hpp"

namespace Poseidon {
namespace Http {

class Client_reader {
private:
	enum State {
		state_first_header      = 0,
		state_headers           = 1,
		state_identity          = 2,
		state_chunk_header      = 3,
		state_chunk_data        = 4,
		state_chunked_trailer   = 5,
	};

protected:
	enum {
		content_length_chunked        =   -1ull,
		content_length_until_eof      =   -2ull,
		content_length_expecting_endl =  -99ull,
		content_length_max            = -100ull,
	};

private:
	Stream_buffer m_queue;

	boost::uint64_t m_size_expecting;
	State m_state;

	Response_headers m_response_headers;
	boost::uint64_t m_content_length;
	boost::uint64_t m_content_offset;

	boost::uint64_t m_chunk_size;
	boost::uint64_t m_chunk_offset;
	Optional_map m_chunked_trailer;

public:
	Client_reader();
	virtual ~Client_reader();

protected:
	// 如果 Transfer-Encoding 为 chunked， content_length 的值为 content_length_chunked。
	virtual void on_response_headers(Response_headers response_headers, boost::uint64_t content_length) = 0;
	// 报文可能分几次收到。
	virtual void on_response_entity(boost::uint64_t entity_offset, Stream_buffer entity) = 0;
	// 报文接收完毕。
	// 如果 on_response_headers() 的 content_length 参数为 content_length_until_eof，此处 real_content_length 即为实际接收大小。
	// 如果 on_response_headers() 的 content_length 参数为 content_length_chunked，使用这个函数标识结束。
	// chunked 允许追加报头。
	virtual bool on_response_end(boost::uint64_t content_length, Optional_map headers) = 0;

public:
	const Stream_buffer &get_queue() const {
		return m_queue;
	}
	Stream_buffer &get_queue(){
		return m_queue;
	}

	bool put_encoded_data(Stream_buffer encoded);

	bool is_content_till_eof() const;
	// 要求 is_content_till_eof() 为 true，否则抛异常。
	bool terminate_content();
};

}
}

#endif
