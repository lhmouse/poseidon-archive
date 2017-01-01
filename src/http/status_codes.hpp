// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2017, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_HTTP_STATUS_CODES_HPP_
#define POSEIDON_HTTP_STATUS_CODES_HPP_

namespace Poseidon {

namespace Http {
	typedef unsigned StatusCode;

	namespace StatusCodes {
		enum {
			// http://www.w3.org/Protocols/rfc2616/rfc2616-sec10.html
			ST_NULL                         =   0,
			ST_CONTINUE                     = 100,
			ST_SWITCHING_PROTOCOLS          = 101,
			ST_OK                           = 200,
			ST_CREATED                      = 201,
			ST_ACCEPTED                     = 202,
			ST_NON_AUTHORITATIVE            = 203,
			ST_NO_CONTENT                   = 204,
			ST_RESET_CONTENT                = 205,
			ST_PARTIAL_CONTENT              = 206,
			ST_MULTIPLE_CHOICES             = 300,
			ST_MOVED_PERMANENTLY            = 301,
			ST_FOUND                        = 302,
			ST_SEE_OTHER                    = 303,
			ST_NOT_MODIFIED                 = 304,
			ST_USE_PROXY                    = 305,
			ST_TEMPORARY_REDIRECT           = 307,
			ST_BAD_REQUEST                  = 400,
			ST_UNAUTHORIZED                 = 401,
			ST_FORBIDDEN                    = 403,
			ST_NOT_FOUND                    = 404,
			ST_METHOD_NOT_ALLOWED           = 405,
			ST_NOT_ACCEPTABLE               = 406,
			ST_PROXY_AUTH_REQUIRED          = 407,
			ST_REQUEST_TIMEOUT              = 408,
			ST_CONFLICT                     = 409,
			ST_GONE                         = 410,
			ST_LENGTH_REQUIRED              = 411,
			ST_PRECONDITION_FAILED          = 412,
			ST_PAYLOAD_TOO_LARGE            = 413,
			ST_URI_TOO_LONG                 = 414,
			ST_UNSUPPORTED_MEDIA_TYPE       = 415,
			ST_RANGE_NOT_SATISFIABLE        = 416,
			ST_EXPECTATION_FAILED           = 417,
			ST_UPGRADE_REQUIRED             = 426,
			ST_INTERNAL_SERVER_ERROR        = 500,
			ST_NOT_IMPLEMENTED              = 501,
			ST_BAD_GATEWAY                  = 502,
			ST_SERVICE_UNAVAILABLE          = 503,
			ST_GATEWAY_TIMEOUT              = 504,
			ST_VERSION_NOT_SUPPORTED        = 505,
		};
	}

	using namespace StatusCodes;

	struct StatusCodeDesc {
		const char *desc_short;
		const char *desc_long;
	};

	extern StatusCodeDesc get_status_code_desc(StatusCode status_code);
}

}

#endif
