// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_CBPP_READER_HPP_
#define POSEIDON_CBPP_READER_HPP_

#include <string>
#include <boost/cstdint.hpp>
#include "../stream_buffer.hpp"
#include "control_codes.hpp"

namespace Poseidon {

namespace Cbpp {
	class Reader {
	private:
		enum State {
			S_PAYLOAD_SIZE		= 0,
			S_EX_PAYLOAD_SIZE	= 1,
			S_MESSAGE_ID		= 2,
			S_DATA_PAYLOAD		= 3,
			S_CONTROL_PAYLOAD	= 4,
		};

	private:
		StreamBuffer m_queue;

		boost::uint64_t m_sizeExpecting;
		State m_state;

		boost::uint64_t m_payloadSize;
		boost::uint16_t m_messageId;
		boost::uint64_t m_payloadOffset;

	public:
		Reader();
		virtual ~Reader();

	protected:
		virtual void onDataMessageHeader(boost::uint16_t messageId, boost::uint64_t payloadSize) = 0;
		virtual void onDataMessagePayload(boost::uint64_t payloadOffset, StreamBuffer payload) = 0;
		// 以下两个回调返回 false 导致于当前消息终止后退出循环。
		virtual bool onDataMessageEnd(boost::uint64_t payloadSize) = 0;

		virtual bool onControlMessage(ControlCode controlCode, boost::int64_t vintParam, std::string stringParam) = 0;

	public:
		const StreamBuffer &getQueue() const {
			return m_queue;
		}
		StreamBuffer &getQueue(){
			return m_queue;
		}

		unsigned getMessageId() const {
			return m_messageId;
		}

		bool putEncodedData(StreamBuffer encoded);
	};
}

}

#endif
