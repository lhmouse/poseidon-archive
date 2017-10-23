// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2017, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_WEBSOCKET_READER_HPP_
#define POSEIDON_WEBSOCKET_READER_HPP_

#include <string>
#include <boost/cstdint.hpp>
#include "../stream_buffer.hpp"
#include "opcodes.hpp"

namespace Poseidon {
namespace WebSocket {

class Reader {
private:
	enum State {
		S_OPCODE            = 0,
		S_FRAME_SIZE        = 1,
		S_FRAME_SIZE_16     = 2,
		S_FRAME_SIZE_64     = 3,
		S_SIZE_END          = 4,
		S_MASK              = 5,
		S_HEADER_END        = 6,
		S_DATA_FRAME        = 7,
		S_CONTROL_FRAME     = 8,
	};

private:
	const bool m_force_masked_frames;

	StreamBuffer m_queue;

	boost::uint64_t m_size_expecting;
	State m_state;

	boost::uint64_t m_whole_offset;
	bool m_prev_fin;

	bool m_fin;
	bool m_masked;
	OpCode m_opcode;
	boost::uint64_t m_frame_size;
	boost::uint32_t m_mask;
	boost::uint64_t m_frame_offset;

public:
	explicit Reader(bool force_masked_frames);
	virtual ~Reader();

protected:
	virtual void on_data_message_header(OpCode opcode) = 0;
	virtual void on_data_message_payload(boost::uint64_t whole_offset, StreamBuffer payload) = 0;
	// 以下两个回调返回 false 导致于当前消息终止后退出循环。
	virtual bool on_data_message_end(boost::uint64_t whole_size) = 0;

	virtual bool on_control_message(OpCode opcode, StreamBuffer payload) = 0;

public:
	const StreamBuffer &get_queue() const {
		return m_queue;
	}
	StreamBuffer &get_queue(){
		return m_queue;
	}

	bool put_encoded_data(StreamBuffer encoded);
};

}
}

#endif
