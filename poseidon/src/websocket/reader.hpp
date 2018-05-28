// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_WEBSOCKET_READER_HPP_
#define POSEIDON_WEBSOCKET_READER_HPP_

#include <string>
#include <boost/cstdint.hpp>
#include "../stream_buffer.hpp"
#include "opcodes.hpp"

namespace Poseidon {
namespace Websocket {

class Reader {
private:
	enum State {
		state_opcode            = 0,
		state_frame_size        = 1,
		state_frame_size_16     = 2,
		state_frame_size_64     = 3,
		state_size_end          = 4,
		state_mask              = 5,
		state_header_end        = 6,
		state_data_frame        = 7,
		state_control_frame     = 8,
	};

private:
	const bool m_force_masked_frames;

	Stream_buffer m_queue;

	std::uint64_t m_size_expecting;
	State m_state;

	std::uint64_t m_whole_offset;
	bool m_prev_fin;

	bool m_fin;
	bool m_masked;
	Opcode m_opcode;
	std::uint64_t m_frame_size;
	std::uint32_t m_mask;
	std::uint64_t m_frame_offset;

public:
	explicit Reader(bool force_masked_frames);
	virtual ~Reader();

protected:
	virtual void on_data_message_header(Opcode opcode) = 0;
	virtual void on_data_message_payload(std::uint64_t whole_offset, Stream_buffer payload) = 0;
	// 以下两个回调返回 false 导致于当前消息终止后退出循环。
	virtual bool on_data_message_end(std::uint64_t whole_size) = 0;

	virtual bool on_control_message(Opcode opcode, Stream_buffer payload) = 0;

public:
	const Stream_buffer & get_queue() const {
		return m_queue;
	}
	Stream_buffer & get_queue(){
		return m_queue;
	}

	bool put_encoded_data(Stream_buffer encoded);
};

}
}

#endif
