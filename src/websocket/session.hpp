// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_WEBSOCKET_SESSION_HPP_
#define POSEIDON_WEBSOCKET_SESSION_HPP_

#include "../cxx_ver.hpp"
#include "../http/upgraded_session_base.hpp"
#include <boost/shared_ptr.hpp>
#include <boost/cstdint.hpp>
#include "../stream_buffer.hpp"
#include "../http/status_codes.hpp"
#include "opcodes.hpp"
#include "status_codes.hpp"

namespace Poseidon {

class OptionalMap;

namespace Http {
	class Header;
	class Session;
}

namespace WebSocket {
	class Session : public Http::UpgradedSessionBase {
	private:
		enum State {
			S_OPCODE			= 0,
			S_PAYLOAD_LEN		= 1,
			S_EX_PAYLOAD_LEN_16	= 2,
			S_EX_PAYLOAD_LEN_64	= 3,
			S_MASK				= 4,
			S_PAYLOAD			= 5,
		};

	private:
		class RequestJob;
		class ErrorJob;

	public:
		static Http::StatusCode makeHttpHandshakeResponse(OptionalMap &ret, const Http::Header &header);

	private:
		StreamBuffer m_received;

		boost::uint64_t m_sizeTotal;
		boost::uint64_t m_sizeExpecting;
		State m_state;

		bool m_fin;
		OpCode m_opcode;
		boost::uint64_t m_payloadLen;
		boost::uint32_t m_payloadMask;

	public:
		Session(const boost::shared_ptr<Http::Session> &parent, std::string uri);
		~Session();

	private:
		void onReadAvail(const void *data, std::size_t size) FINAL;

		bool sendFrame(StreamBuffer payload, OpCode opcode, bool fin, bool masked);

	protected:
		virtual void onRequest(OpCode opcode, const StreamBuffer &payload) = 0;

	public:
		bool send(StreamBuffer payload, bool binary = true, bool fin = false, bool masked = false);
		bool shutdown(StatusCode statusCode, StreamBuffer additional = StreamBuffer()) NOEXCEPT;
	};
}

}

#endif
