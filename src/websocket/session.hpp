// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_WEBSOCKET_SESSION_HPP_
#define POSEIDON_WEBSOCKET_SESSION_HPP_

#include "../http/upgraded_session_base.hpp"
#include "opcodes.hpp"
#include "status_codes.hpp"
#include "reader.hpp"
#include "writer.hpp"

namespace Poseidon {

namespace WebSocket {
	class Session : public Http::UpgradedSessionBase, private Reader, private Writer {
	private:
		class SyncJobBase;
		class DataMessageJob;
		class ControlMessageJob;
		class ErrorJob;

	private:
		const boost::uint64_t m_maxRequestLength;

		boost::uint64_t m_sizeTotal;
		OpCode m_opcode;
		StreamBuffer m_payload;

	public:
		explicit Session(const boost::shared_ptr<Http::Session> &parent, boost::uint64_t maxRequestLength = 0);
		~Session();

	protected:
		// UpgradedSessionBase
		void onReadAvail(StreamBuffer data) OVERRIDE;

		// Reader
		void onDataMessageHeader(OpCode opcode) OVERRIDE;
		void onDataMessagePayload(boost::uint64_t wholeOffset, StreamBuffer payload) OVERRIDE;
		bool onDataMessageEnd(boost::uint64_t wholeSize) OVERRIDE;

		bool onControlMessage(OpCode opcode, StreamBuffer payload) OVERRIDE;

		// Writer
		long onEncodedDataAvail(StreamBuffer encoded) OVERRIDE;

		// 可覆写。
		virtual void onSyncDataMessage(OpCode opcode, const StreamBuffer &payload) = 0;

		virtual void onSyncControlMessage(OpCode opcode, const StreamBuffer &payload);

	public:
		bool shutdownRead() NOEXCEPT OVERRIDE {
			return shutdown(ST_NORMAL_CLOSURE);
		}
		bool shutdownWrite() NOEXCEPT OVERRIDE {
			return shutdown(ST_NORMAL_CLOSURE);
		}

		bool send(StreamBuffer payload, bool binary, bool masked = false);

		bool shutdown(StatusCode statusCode, StreamBuffer additional = StreamBuffer()) NOEXCEPT;
	};
}

}

#endif
