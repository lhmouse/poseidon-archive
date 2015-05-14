// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_CBPP_SESSION_HPP_
#define POSEIDON_CBPP_SESSION_HPP_

#include "../tcp_session_base.hpp"
#include "reader.hpp"
#include "control_codes.hpp"
#include "status_codes.hpp"

namespace Poseidon {

namespace Cbpp {
	class Session : public TcpSessionBase, private Reader {
	private:
		class DataMessageJob;

		class ControlMessageJob;

		class ErrorJob;

	private:
		boost::uint64_t m_sizeTotal;
		unsigned m_messageId;
		StreamBuffer m_payload;

	public:
		explicit Session(UniqueFile socket);
		~Session();

	protected:
		// TcpSessionBase
		void onReadAvail(const void *data, std::size_t size) OVERRIDE;

		// Reader
		void onDataMessageHeader(boost::uint16_t messageId, boost::uint64_t payloadSize) OVERRIDE;
		void onDataMessagePayload(boost::uint64_t payloadOffset, StreamBuffer payload) OVERRIDE;
		bool onDataMessageEnd(boost::uint64_t payloadSize) OVERRIDE;

		bool onControlMessage(ControlCode controlCode, boost::int64_t vintParam, std::string stringParam) OVERRIDE;

		// 可覆写。
		virtual void onSyncDataMessage(boost::uint16_t messageId, const StreamBuffer &payload) = 0;

		virtual void onSyncControlMessage(ControlCode controlCode, boost::int64_t vintParam, const std::string &stringParam);

	public:
		bool send(boost::uint16_t messageId, StreamBuffer payload);
		bool sendError(boost::uint16_t messageId, StatusCode statusCode, std::string reason);

		template<typename MsgT>
		bool send(const MsgT &msg){
			return send(MsgT::ID, StreamBuffer(msg));
		}
	};
}

}

#endif
