// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_HTTP_STATUS_CODES_HPP_
#define POSEIDON_HTTP_STATUS_CODES_HPP_

namespace Poseidon {
namespace Http {

typedef unsigned Status_code;

namespace Status_codes {
	enum {
		// http://www.w3.org/Protocols/rfc2616/rfc2616-sec10.html
		static_invalid                      =  -1,
		status_continue                     = 100,
		status_switching_protocols          = 101,
		status_ok                           = 200,
		status_created                      = 201,
		status_accepted                     = 202,
		status_non_authoritative            = 203,
		status_no_content                   = 204,
		status_reset_content                = 205,
		status_partial_content              = 206,
		status_multiple_choices             = 300,
		status_moved_permanently            = 301,
		status_found                        = 302,
		status_see_other                    = 303,
		status_not_modified                 = 304,
		status_use_proxy                    = 305,
		status_temporary_redirect           = 307,
		status_bad_request                  = 400,
		status_unauthorized                 = 401,
		status_forbidden                    = 403,
		status_not_found                    = 404,
		status_method_not_allowed           = 405,
		status_not_acceptable               = 406,
		status_proxy_auth_required          = 407,
		status_request_timeout              = 408,
		status_conflict                     = 409,
		status_gone                         = 410,
		status_length_required              = 411,
		status_precondition_failed          = 412,
		status_payload_too_large            = 413,
		status_uri_too_long                 = 414,
		status_unsupported_media_type       = 415,
		status_range_not_satisfiable        = 416,
		status_expectation_failed           = 417,
		status_upgrade_required             = 426,
		status_internal_server_error        = 500,
		status_not_implemented              = 501,
		status_bad_gateway                  = 502,
		status_service_unavailable          = 503,
		status_gateway_timeout              = 504,
		status_version_not_supported        = 505,
	};
}

using namespace Status_codes;

struct Status_code_desc {
	const char *desc_short;
	const char *desc_long;
};

extern Status_code_desc get_status_code_desc(Status_code status_code) noexcept;

}
}

#endif
