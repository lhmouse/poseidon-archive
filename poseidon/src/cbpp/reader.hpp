// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_CBPP_READER_HPP_
#define POSEIDON_CBPP_READER_HPP_

#include <string>
#include <boost/cstdint.hpp>
#include "../stream_buffer.hpp"
#include "status_codes.hpp"

namespace Poseidon {
namespace Cbpp {

class Reader {
private:
	enum State {
		state_payload_size      = 0,
		state_ex_payload_size   = 1,
		state_message_id        = 2,
		state_data_payload      = 3,
		state_control_payload   = 4,
	};

private:
	Stream_buffer m_queue;

	boost::uint64_t m_size_expecting;
	State m_state;

	boost::uint64_t m_payload_size;
	boost::uint16_t m_message_id;
	boost::uint64_t m_payload_offset;

public:
	Reader();
	virtual ~Reader();

protected:
	virtual void on_data_message_header(boost::uint16_t message_id, boost::uint64_t payload_size) = 0;
	virtual void on_data_message_payload(boost::uint64_t payload_offset, Stream_buffer payload) = 0;
	// 以下两个回调返回 false 导致于当前消息终止后退出循环。
	virtual bool on_data_message_end(boost::uint64_t payload_size) = 0;

	virtual bool on_control_message(Status_code status_code, Stream_buffer param) = 0;

public:
	const Stream_buffer &get_queue() const {
		return m_queue;
	}
	Stream_buffer &get_queue(){
		return m_queue;
	}

	unsigned get_message_id() const {
		return m_message_id;
	}

	bool put_encoded_data(Stream_buffer encoded);
};

}
}

#endif
