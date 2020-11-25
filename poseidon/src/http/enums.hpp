// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_HTTP_ENUMS_HPP_
#define POSEIDON_HTTP_ENUMS_HPP_

#include "../fwd.hpp"

namespace poseidon {

// These are HTTP version numbers.
// At the moment only 1.0 and 1.1 are supported.
enum HTTP_Version : uint16_t
  {
    http_version_0_0  =      0,
    http_version_1_0  = 0x0100,  // HTTP/1.0
    http_version_1_1  = 0x0101,  // HTTP/1.1
  };

// Converts an HTTP version to a string such as `HTTP/1.1`.
ROCKET_CONST_FUNCTION
const char*
format_http_version(HTTP_Version ver)
  noexcept;

// Parses a version number from plain text.
// `http_version_0_0` is returned if the string is not valid.
ROCKET_PURE_FUNCTION
HTTP_Version
parse_http_version(const char* bptr, const char* eptr)
  noexcept;

ROCKET_PURE_FUNCTION inline
HTTP_Version
parse_http_version(const char* bptr, size_t len)
  noexcept
  { return noadl::parse_http_version(bptr, bptr + len);  }

ROCKET_PURE_FUNCTION inline
HTTP_Version
parse_http_version(const char* bptr)
  noexcept
  { return noadl::parse_http_version(bptr, ::std::strlen(bptr));  }

// These are HTTP verbs a.k.a. methods.
enum HTTP_Verb : uint8_t
  {
    http_verb_NULL     = 0,
    http_verb_GET      = 1,
    http_verb_HEAD     = 2,
    http_verb_POST     = 3,
    http_verb_PUT      = 4,
    http_verb_DELETE   = 5,
    http_verb_CONNECT  = 6,
    http_verb_OPTIONS  = 7,
    http_verb_TRACE    = 8,
  };

// Converts an HTTP verb to a string such as `GET`.
// If the verb is invalid, the invalid string `NULL` is returned.
ROCKET_CONST_FUNCTION
const char*
format_http_verb(HTTP_Verb verb)
  noexcept;

// Parses a verb from plain text.
// `http_verb_null` is returned if the string is not valid.
ROCKET_PURE_FUNCTION
HTTP_Verb
parse_http_verb(const char* bptr, const char* eptr)
  noexcept;

ROCKET_PURE_FUNCTION inline
HTTP_Verb
parse_http_verb(const char* bptr, size_t len)
  noexcept
  { return noadl::parse_http_verb(bptr, bptr + len);  }

ROCKET_PURE_FUNCTION inline
HTTP_Verb
parse_http_verb(const char* bptr)
  noexcept
  { return noadl::parse_http_verb(bptr, ::std::strlen(bptr));  }

// These are HTTP status codes.
// This list is not exhaustive. Custom values may be used.
enum HTTP_Status : uint16_t
  {
    http_status_null                   =   0,  // Null
    http_status_100                    = 100,
    http_status_continue               = 100,  // Continue
    http_status_switching_protocol     = 101,  // Switching Protocol
    http_status_processing             = 102,  // Processing
    http_status_early_hints            = 103,  // Early Hints
    http_status_200                    = 200,
    http_status_ok                     = 200,  // OK
    http_status_created                = 201,  // Created
    http_status_accepted               = 202,  // Accepted
    http_status_nonauthoritative       = 203,  // Non-authoritative Information
    http_status_no_content             = 204,  // No Content
    http_status_reset_content          = 205,  // Reset Content
    http_status_partial_content        = 206,  // Partial Content
    http_status_multistatus            = 207,  // Multi-status
    http_status_already_reported       = 208,  // Already Reported
    http_status_im_used                = 226,  // IM Used
    http_status_300                    = 300,
    http_status_multiple_choice        = 300,  // Multiple Choice
    http_status_moved_permanently      = 301,  // Moved Permanently
    http_status_found                  = 302,  // Found
    http_status_see_other              = 303,  // See Other
    http_status_not_modified           = 304,  // Not Modified
    http_status_use_proxy              = 305,  // Use Proxy
    http_status_temporary_redirect     = 307,  // Temporary Redirect
    http_status_permanent_redirect     = 308,  // Permanent Redirect
    http_status_400                    = 400,
    http_status_bad_request            = 400,  // Bad Request
    http_status_unauthorized           = 401,  // Unauthorized
    http_status_forbidden              = 403,  // Forbidden
    http_status_not_found              = 404,  // Not Found
    http_status_method_not_allowed     = 405,  // Method Not Allowed
    http_status_not_acceptable         = 406,  // Not Acceptable
    http_status_proxy_unauthorized     = 407,  // Proxy Authentication Required
    http_status_request_timedout       = 408,  // Request Timeout
    http_status_conflict               = 409,  // Conflict
    http_status_gone                   = 410,  // Gone
    http_status_length_required        = 411,  // Length Required
    http_status_precondition_failed    = 412,  // Precondition Failed
    http_status_payload_too_large      = 413,  // Payload Too Large
    http_status_uri_too_long           = 414,  // URI Too Long
    http_status_unsupported_media      = 415,  // Unsupported Media Type
    http_status_range_not_satisfiable  = 416,  // Range Not Satisfiable
    http_status_expectation_failed     = 417,  // Expectation Failed
    http_status_misdirected_request    = 421,  // Misdirected Request
    http_status_unprocessable          = 422,  // Unprocessable Entity
    http_status_locked                 = 423,  // Locked
    http_status_failed_dependency      = 424,  // Failed Dependency
    http_status_too_early              = 425,  // Too Early
    http_status_upgrade_required       = 426,  // Upgrade Required
    http_status_precondition_required  = 428,  // Precondition Required
    http_status_too_many_requests      = 429,  // Too Many Requests
    http_status_headers_too_large      = 431,  // Request Header Fields Too Large
    http_status_500                    = 500,
    http_status_internal_server_error  = 500,  // Internal Server Error
    http_status_not_implemented        = 501,  // Not Implemented
    http_status_bad_gateway            = 502,  // Bad Gateway
    http_status_service_unavailable    = 503,  // Service Unavailable
    http_status_gateway_timeout        = 504,  // Gateway Timeout
    http_status_version_not_supported  = 505,  // HTTP Version Not Supported
    http_status_insufficient_storage   = 507,  // Insufficient Storage
    http_status_loop_detected          = 508,  // Loop Detected
    http_status_not_extended           = 510,  // Not Extended
    http_status_network_unauthorized   = 511,  // Network Authentication Required
  };

// Converts an HTTP status code to a string such as `Bad Request`.
// If the status code is unknown, `Unknown Status` is returned.
ROCKET_CONST_FUNCTION
const char*
describe_http_status(HTTP_Status stat)
  noexcept;

// These are HTTP status code classes.
enum HTTP_Status_Class : uint8_t
  {
    http_status_class_null           = 0,
    http_status_class_informational  = 1,
    http_status_class_successful     = 2,
    http_status_class_redirection    = 3,
    http_status_class_client_error   = 4,
    http_status_class_server_error   = 5,
  };

// Classifies a status code.
constexpr
HTTP_Status_Class
classify_http_status(HTTP_Status stat)
  noexcept
  { return static_cast<HTTP_Status_Class>(static_cast<uint32_t>(stat) / 100);  }

// Gets the x00 status code.
constexpr
HTTP_Status
get_http_status_base(HTTP_Status_Class stcls)
  noexcept
  { return static_cast<HTTP_Status>(static_cast<uint32_t>(stcls) * 100);  }

}  // namespace poseidon

#endif
