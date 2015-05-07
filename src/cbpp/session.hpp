// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_CBPP_SESSION_HPP_
#define POSEIDON_CBPP_SESSION_HPP_

#include "low_level_session.hpp"

namespace Poseidon {

namespace Cbpp {
	class Session : public LowLevelSession {
	private:
		class RequestJob;
		class ControlJob;

		class ErrorJob;

	public:
		explicit Session(UniqueFile socket);
		~Session();

	protected:
		void onLowLevelRequest(boost::uint16_t messageId, StreamBuffer payload) OVERRIDE;
		void onLowLevelControl(ControlCode controlCode, boost::int64_t intParam, std::string strParam) OVERRIDE;

		void onLowLevelError(unsigned messageId, StatusCode statusCode, const char *reason) OVERRIDE;

		virtual void onRequest(boost::uint16_t messageId, const StreamBuffer &payload) = 0;
		virtual void onControl(ControlCode controlCode, boost::int64_t intParam, const std::string &strParam);
	};
}

}

#endif
