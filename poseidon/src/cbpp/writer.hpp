// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_CBPP_WRITER_HPP_
#define POSEIDON_CBPP_WRITER_HPP_

#include <string>
#include <boost/cstdint.hpp>
#include "../stream_buffer.hpp"
#include "status_codes.hpp"

namespace Poseidon {
namespace Cbpp {

class Writer {
public:
	Writer();
	virtual ~Writer();

protected:
	virtual long on_encoded_data_avail(Stream_buffer encoded) = 0;

public:
	long put_data_message(std::uint16_t message_id, Stream_buffer payload);
	long put_control_message(Status_code status_code, Stream_buffer param);
};

}
}

#endif
