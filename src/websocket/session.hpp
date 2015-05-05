// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_WEBSOCKET_SESSION_HPP_
#define POSEIDON_WEBSOCKET_SESSION_HPP_

#include "low_level_session.hpp"

namespace Poseidon {

namespace WebSocket {
	class Session : public LowLevelSession {
	private:
		class RequestJob;
		class ErrorJob;

	public:
		Session(const boost::shared_ptr<Http::LowLevelSession> &parent, std::string uri);
		~Session();

	protected:
		void onLowLevelRequest(OpCode opcode, StreamBuffer payload) OVERRIDE;
		void onLowLevelError(StatusCode statusCode, const char *reason) OVERRIDE;

		virtual void onRequest(OpCode opcode, const StreamBuffer &payload) = 0;
	};
}

}

#endif
