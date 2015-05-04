// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_CBPP_CLIENT_HPP_
#define POSEIDON_CBPP_CLIENT_HPP_

#include "../tcp_client_base.hpp"
#include <string>
#include <boost/shared_ptr.hpp>
#include <boost/type_traits/is_base_of.hpp>
#include <boost/utility/enable_if.hpp>
#include <boost/cstdint.hpp>
#include "../stream_buffer.hpp"
#include "message_base.hpp"
#include "control_codes.hpp"

namespace Poseidon {

class TimerItem;

namespace Cbpp {
	class Client : public TcpClientBase {
	private:
		enum State {
			S_PAYLOAD_LEN		= 0,
			S_EX_PAYLOAD_LEN	= 1,
			S_MESSAGE_ID		= 2,
			S_PAYLOAD			= 3,
		};

	private:
		class KeepAliveJob;

		class ResponseJob;
		class PayloadJob;
		class ControlJob;

	private:
		const boost::uint64_t m_keepAliveTimeout;

		boost::shared_ptr<const TimerItem> m_keepAliveTimer;

		StreamBuffer m_received;

		boost::uint64_t m_sizeExpecting;
		State m_state;

		boost::uint64_t m_payloadLen;
		boost::uint16_t m_messageId;
		boost::uint64_t m_payloadOffset;

	protected:
		Client(const IpPort &addr, boost::uint64_t keepAliveTimeout, bool useSsl);
		~Client();

	protected:
		void onReadAvail(const void *data, std::size_t size) OVERRIDE;

		virtual void onResponse(boost::uint16_t messageId, boost::uint64_t payloadLen) = 0;
		// 报文可能分几次收到。
		virtual void onPayload(boost::uint64_t payloadOffset, const StreamBuffer &payload) = 0;

		virtual void onControl(boost::uint16_t messageId, StatusCode statusCode, const std::string &reason) = 0;

	public:
		bool send(boost::uint16_t messageId, StreamBuffer payload);

		template<class MessageT>
		typename boost::enable_if<boost::is_base_of<MessageBase, MessageT>, bool>::type send(const MessageT &payload){
			return send(MessageT::ID, StreamBuffer(payload));
		}
	};
}

}

#endif
