// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2016, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_HTTP_EXCEPTION_HPP_
#define POSEIDON_HTTP_EXCEPTION_HPP_

#include "../protocol_exception.hpp"
#include "../optional_map.hpp"
#include "status_codes.hpp"

namespace Poseidon {

namespace Http {
	class Exception : public ProtocolException {
	private:
		boost::shared_ptr<OptionalMap> m_headers;

	public:
		// message 设定为首字节为 0xFF 则发送默认页面。
		Exception(const char *file, std::size_t line, StatusCode status_code,
			OptionalMap headers = OptionalMap(), SharedNts message = sslit("\xFF"));
		Exception(const char *file, std::size_t line, StatusCode status_code, SharedNts message);
		~Exception() NOEXCEPT;

	public:
		StatusCode status_code() const NOEXCEPT {
			return static_cast<StatusCode>(code());
		}
		const OptionalMap &headers() const NOEXCEPT;
	};
}

}

#endif
