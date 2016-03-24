// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2016, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_HTTP_SESSION_HPP_
#define POSEIDON_HTTP_SESSION_HPP_

#include "low_level_session.hpp"

namespace Poseidon {

namespace Http {
	class Session : public LowLevelSession {
	private:
		class SyncJobBase;
		class ContinueJob;
		class RequestJob;
		class ErrorJob;

	public:
		explicit Session(UniqueFile socket, boost::uint64_t max_request_length = 0);
		~Session();

	protected:
		// LowLevelSession
		boost::shared_ptr<UpgradedSessionBase> on_low_level_request(
			RequestHeaders request_headers, std::string transfer_encoding, StreamBuffer entity) OVERRIDE;

		// 可覆写。
		virtual void on_sync_request(RequestHeaders request_headers, StreamBuffer entity) = 0;
	};
}

}

#endif
