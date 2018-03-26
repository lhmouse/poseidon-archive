// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_HTTP_CLIENT_WRITER_HPP_
#define POSEIDON_HTTP_CLIENT_WRITER_HPP_

#include <string>
#include <cstddef>
#include <boost/cstdint.hpp>
#include "../stream_buffer.hpp"
#include "../optional_map.hpp"
#include "request_headers.hpp"

namespace Poseidon {
namespace Http {

class Client_writer {
public:
	Client_writer();
	virtual ~Client_writer();

protected:
	virtual long on_encoded_data_avail(Stream_buffer encoded) = 0;

public:
	long put_request(Request_headers request_headers, Stream_buffer entity, bool set_content_length);

	long put_chunked_header(Request_headers request_headers);
	long put_chunk(Stream_buffer entity);
	long put_chunked_trailer(Optional_map headers);
};

}
}

#endif
