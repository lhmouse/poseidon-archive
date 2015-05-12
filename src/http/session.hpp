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
		// transferEncoding 确保已被转换为小写、已排序，并且 identity 被移除（如果有的话）。
		boost::shared_ptr<UpgradedLowLevelSessionBase> onLowLevelRequestHeaders(RequestHeaders &requestHeaders,
			const std::vector<std::string> &transferEncoding, boost::uint64_t contentLength) OVERRIDE;

		void onLowLevelRequest(RequestHeaders requestHeaders, StreamBuffer entity) OVERRIDE;
		void onLowLevelError(StatusCode statusCode, OptionalMap headers) OVERRIDE;

		virtual void onRequest(const RequestHeaders &requestHeaders, const StreamBuffer &entity) = 0;
	};
}

}

#endif
