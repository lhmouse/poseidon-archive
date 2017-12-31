// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "exception.hpp"
#include "../log.hpp"

namespace Poseidon {
namespace Http {

namespace {
	const OptionalMap g_empty_heades;
}

Exception::Exception(const char *file, std::size_t line, const char *func, StatusCode status_code, OptionalMap headers)
	: ProtocolException(file, line, func, SharedNts::view(get_status_code_desc(status_code).desc_short), static_cast<long>(status_code))
	, m_headers(!headers.empty() ? boost::make_shared<OptionalMap>(STD_MOVE(headers)) : boost::shared_ptr<OptionalMap>())
{
	LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "Http::Exception: code = ", get_code(), ", what = ", what());
}
Exception::~Exception() NOEXCEPT { }

const OptionalMap &Exception::get_headers() const NOEXCEPT {
	return m_headers ? *m_headers : g_empty_heades;
}

}
}
