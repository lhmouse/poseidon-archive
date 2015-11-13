// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_CBPP_WRITER_HPP_
#define POSEIDON_CBPP_WRITER_HPP_

#include <string>
#include <boost/cstdint.hpp>
#include "../stream_buffer.hpp"
#include "control_codes.hpp"

namespace Poseidon {

namespace Cbpp {
	class Writer {
	public:
		Writer();
		virtual ~Writer();

	protected:
		virtual long on_encoded_data_avail(StreamBuffer encoded) = 0;

	public:
		long put_data_message(boost::uint16_t message_id, StreamBuffer payload);

		long put_control_message(ControlCode control_code, boost::int64_t vint_param, std::string string_param);
	};
}

}

#endif
