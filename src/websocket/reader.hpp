// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

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
			S_OPCODE			= 0,
			S_FRAME_SIZE		= 1,
			S_FRAME_SIZE_16		= 2,
			S_FRAME_SIZE_64		= 3,
			S_MASK				= 4,
			S_DATA_FRAME		= 5,
			S_CONTROL_FRAME		= 6,
		};

	private:
		StreamBuffer m_queue;

		boost::uint64_t m_sizeExpecting;
		State m_state;

		boost::uint64_t m_wholeOffset;
		bool m_prevFin;

		bool m_fin;
		OpCode m_opcode;
		boost::uint64_t m_frameSize;
		boost::uint32_t m_mask;
		boost::uint64_t m_frameOffset;

	public:
		Reader();
		virtual ~Reader();

	protected:
		virtual void onDataMessageHeader(OpCode opcode) = 0;
		virtual void onDataMessagePayload(boost::uint64_t wholeOffset, StreamBuffer payload) = 0;
		// 以下两个回调返回 false 导致于当前消息终止后退出循环。
		virtual bool onDataMessageEnd(boost::uint64_t wholeSize) = 0;

		virtual bool onControlMessage(OpCode opcode, StreamBuffer payload) = 0;

	public:
		bool putEncodedData(StreamBuffer encoded);
	};
}

}

#endif
