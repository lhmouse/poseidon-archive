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
#include "../http/verbs.hpp"
#include "opcodes.hpp"
#include "status_codes.hpp"

namespace Poseidon {

class OptionalMap;

namespace Http {
	class Session;
}

namespace WebSocket {
	class Session : public Http::UpgradedSessionBase {
	private:
		enum State {
			S_OPCODE,
			S_PAYLOAD_LEN,
			S_EX_PAYLOAD_LEN,
			S_MASK,
			S_PAYLOAD,
		};

	private:
		class RequestJob;
		class ErrorJob;

	public:
		static Http::StatusCode makeHttpHandshakeResponse(OptionalMap &ret,
			Http::Verb verb, unsigned version, const OptionalMap &headers);

	private:
		State m_state;
		bool m_fin;
		OpCode m_opcode;
		boost::uint64_t m_payloadLen;
		boost::uint32_t m_payloadMask;
		StreamBuffer m_payload;
		StreamBuffer m_whole;

	public:
		Session(const boost::shared_ptr<Http::Session> &parent, std::string uri);

	private:
		void onReadAvail(const void *data, std::size_t size) OVERRIDE FINAL;

		void onControlFrame();

		bool sendFrame(StreamBuffer contents, OpCode opcode, bool fin, bool masked);

	protected:
		virtual void onRequest(OpCode opcode, const StreamBuffer &contents) = 0;

	public:
		bool send(StreamBuffer contents, bool binary = true, bool fin = false, bool masked = false);
		bool shutdown(StatusCode statusCode, StreamBuffer additional = StreamBuffer()) NOEXCEPT;
	};
}

}

#endif
