// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2016, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "exception.hpp"
#include "../log.hpp"

namespace Poseidon {

namespace {
	const OptionalMap EMPTY_HEADERS;

	SharedNts replace_with_default(SharedNts message){
		if(message[0] == (char)0xFF){
			message = sslit("\xFF     <Use default HTML page>");
		}
		return STD_MOVE(message);
	}
}

namespace Http {
	Exception::Exception(const char *file, std::size_t line, const char *func, StatusCode status_code, OptionalMap headers, SharedNts message)
		: ProtocolException(file, line, func, replace_with_default(STD_MOVE(message)), static_cast<long>(status_code))
	{
		if(!headers.empty()){
			m_headers = boost::make_shared<OptionalMap>(STD_MOVE(headers));
		}

		LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO,
			"Http::Exception: status_code = ", status_code, ", what = ", what());
	}
	Exception::Exception(const char *file, std::size_t line, const char *func, StatusCode status_code, SharedNts message)
		: ProtocolException(file, line, func, replace_with_default(STD_MOVE(message)), static_cast<long>(status_code))
	{
		LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO,
			"Http::Exception: status_code = ", status_code, ", what = ", what());
	}
	Exception::~Exception() NOEXCEPT {
	}

	const OptionalMap &Exception::headers() const NOEXCEPT {
		return m_headers ? *m_headers : EMPTY_HEADERS;
	}
}

}
