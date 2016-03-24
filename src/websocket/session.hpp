// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2016, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_WEBSOCKET_SESSION_HPP_
#define POSEIDON_WEBSOCKET_SESSION_HPP_

#include "low_level_session.hpp"

namespace Poseidon {

namespace WebSocket {
	class Session : public LowLevelSession {
	private:
		class SyncJobBase;
		class DataMessageJob;
		class ControlMessageJob;

	public:
		explicit Session(const boost::shared_ptr<Http::LowLevelSession> &parent, boost::uint64_t max_request_length = 0);
		~Session();

	protected:
		bool on_low_level_data_message(OpCode opcode, StreamBuffer payload) OVERRIDE;
		bool on_low_level_control_message(OpCode opcode, StreamBuffer payload) OVERRIDE;

		// 可覆写。
		virtual void on_sync_data_message(OpCode opcode, StreamBuffer payload) = 0;
		virtual void on_sync_control_message(OpCode opcode, StreamBuffer payload);
	};
}

}

#endif
