// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_WEBSOCKET_WRITER_HPP_
#define POSEIDON_WEBSOCKET_WRITER_HPP_

#include <string>
#include <boost/cstdint.hpp>
#include "status_codes.hpp"
#include "../stream_buffer.hpp"

namespace Poseidon {
namespace Websocket {

class Writer {
public:
	Writer();
	virtual ~Writer();

protected:
	virtual long on_encoded_data_avail(Stream_buffer encoded) = 0;

public:
	long put_message(int opcode, bool masked, Stream_buffer payload);
	long put_close_message(Status_code status_code, bool masked, Stream_buffer addition);
};

}
}

#endif
