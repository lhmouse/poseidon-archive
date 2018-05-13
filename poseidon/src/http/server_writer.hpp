// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_HTTP_SERVER_WRITER_HPP_
#define POSEIDON_HTTP_SERVER_WRITER_HPP_

#include <string>
#include <cstddef>
#include <boost/cstdint.hpp>
#include "../stream_buffer.hpp"
#include "../option_map.hpp"
#include "response_headers.hpp"

namespace Poseidon {
namespace Http {

class Server_writer {
public:
	Server_writer();
	virtual ~Server_writer();

protected:
	virtual long on_encoded_data_avail(Stream_buffer encoded) = 0;

public:
	long put_response(Response_headers response_headers, Stream_buffer entity, bool set_content_length);

	long put_chunked_header(Response_headers response_headers);
	long put_chunk(Stream_buffer entity);
	long put_chunked_trailer(Option_map headers);
};

}
}

#endif
