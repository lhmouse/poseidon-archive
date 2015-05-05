// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_HTTP_SESSION_HPP_
#define POSEIDON_HTTP_SESSION_HPP_

#include "low_level_session.hpp"

namespace Poseidon {

namespace Http {
	class Session : public LowLevelSession {
	private:
		class ContinueJob;
		class RequestJob;
		class ErrorJob;

	public:
		explicit Session(UniqueFile socket);
		~Session();

	protected:
		boost::shared_ptr<UpgradedLowLevelSessionBase> onLowLevelRequestHeaders(
			RequestHeaders &requestHeaders, boost::uint64_t contentLength) OVERRIDE;

		void onLowLevelRequest(RequestHeaders requestHeaders, StreamBuffer entity) OVERRIDE;
		void onLowLevelError(StatusCode statusCode, OptionalMap headers) OVERRIDE;

		virtual void onRequest(const RequestHeaders &requestHeaders, const StreamBuffer &entity) = 0;
	};
}

}

#endif
