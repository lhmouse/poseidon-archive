// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "exception.hpp"
using namespace Poseidon;

namespace {

const OptionalMap EMPTY_HEADERS;

}

HttpException::HttpException(const char *file, std::size_t line,
	HttpStatus status, OptionalMap headers)
	: ProtocolException(file, line,
		getHttpStatusDesc(status).descShort, static_cast<unsigned>(status))
{
	if(!headers.empty()){
		boost::make_shared<OptionalMap>().swap(m_headers);
		m_headers->swap(headers);
	}
}
HttpException::~HttpException() NOEXCEPT {
}

const OptionalMap &HttpException::headers() const NOEXCEPT {
	return m_headers ? *m_headers : EMPTY_HEADERS;
}
