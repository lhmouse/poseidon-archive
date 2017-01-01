// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2017, LH_Mouse. All wrongs reserved.

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
			S_PAYLOAD_SIZE      = 0,
			S_EX_PAYLOAD_SIZE   = 1,
			S_MESSAGE_ID        = 2,
			S_DATA_PAYLOAD      = 3,
			S_CONTROL_PAYLOAD   = 4,
		};

	private:
		StreamBuffer m_queue;

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
		virtual void on_data_message_payload(boost::uint64_t payload_offset, StreamBuffer payload) = 0;
		// 以下两个回调返回 false 导致于当前消息终止后退出循环。
		virtual bool on_data_message_end(boost::uint64_t payload_size) = 0;

		virtual bool on_control_message(StatusCode status_code, StreamBuffer param) = 0;

	public:
		const StreamBuffer &get_queue() const {
			return m_queue;
		}
		StreamBuffer &get_queue(){
			return m_queue;
		}

		unsigned get_message_id() const {
			return m_message_id;
		}

		bool put_encoded_data(StreamBuffer encoded);
	};
}

}

#endif
