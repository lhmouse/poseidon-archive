// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_CBPP_CLIENT_HPP_
#define POSEIDON_CBPP_CLIENT_HPP_

#include "low_level_client.hpp"

namespace Poseidon {

namespace Cbpp {
	class Client : public LowLevelClient {
	private:
		class ResponseJob;
		class PayloadJob;
		class ControlJob;

	protected:
		Client(const IpPort &addr, boost::uint64_t keepAliveTimeout, bool useSsl);
		~Client();

	protected:
		void onLowLevelResponse(boost::uint16_t messageId, boost::uint64_t payloadLen) OVERRIDE;
		void onLowLevelPayload(boost::uint64_t payloadOffset, StreamBuffer payload) OVERRIDE;

		void onLowLevelControl(boost::uint16_t messageId, StatusCode statusCode, std::string reason) OVERRIDE;

		virtual void onResponse(boost::uint16_t messageId, boost::uint64_t payloadLen) = 0;
		// 报文可能分几次收到。
		virtual void onPayload(boost::uint64_t payloadOffset, const StreamBuffer &payload) = 0;

		virtual void onControl(boost::uint16_t messageId, StatusCode statusCode, const std::string &reason) = 0;
	};
}

}

#endif
