// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_CBPP_LOW_LEVEL_SESSION_HPP_
#define POSEIDON_CBPP_LOW_LEVEL_SESSION_HPP_

#include <cstddef>
#include <boost/type_traits/is_base_of.hpp>
#include <boost/utility/enable_if.hpp>
#include <boost/cstdint.hpp>
#include "../tcp_session_base.hpp"
#include "../stream_buffer.hpp"
#include "control_codes.hpp"
#include "status_codes.hpp"

namespace Poseidon {

namespace Cbpp {
	class MessageBase;

	class LowLevelSession : public TcpSessionBase {
	private:
		enum State {
			S_PAYLOAD_LEN		= 0,
			S_EX_PAYLOAD_LEN	= 1,
			S_MESSAGE_ID		= 2,
			S_PAYLOAD			= 3,
		};

	private:
		StreamBuffer m_received;

		boost::uint64_t m_sizeTotal;
		boost::uint64_t m_sizeExpecting;
		State m_state;

		boost::uint16_t m_messageId;
		boost::uint64_t m_payloadLen;

	public:
		explicit LowLevelSession(UniqueFile socket);
		~LowLevelSession();

	protected:
		void onReadAvail(const void *data, std::size_t size) OVERRIDE;

		virtual void onLowLevelRequest(boost::uint16_t messageId, StreamBuffer payload) = 0;
		virtual void onLowLevelControl(ControlCode controlCode, boost::int64_t intParam, std::string strParam) = 0;

		virtual void onLowLevelError(unsigned messageId, StatusCode statusCode, const char *reason) = 0;

	public:
		bool send(boost::uint16_t messageId, StreamBuffer payload);

		template<class MessageT>
		typename boost::enable_if<boost::is_base_of<MessageBase, MessageT>, bool>::type
			send(const MessageT &payload)
		{
			return send(MessageT::ID, StreamBuffer(payload));
		}

		bool sendError(boost::uint16_t messageId, StatusCode statusCode, std::string reason);
	};
}

}

#endif
