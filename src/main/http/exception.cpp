#include "../precompiled.hpp"
#include "exception.hpp"
using namespace Poseidon;

HttpException::HttpException(const char *file, std::size_t line,
	HttpStatus status, OptionalMap headers)
	: ProtocolException(file, line,
		getHttpStatusDesc(status).descShort, static_cast<unsigned>(status))
	, m_headers(STD_MOVE(headers))
{
}
HttpException::~HttpException() NOEXCEPT {
}
