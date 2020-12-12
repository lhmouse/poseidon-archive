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

// These are HTTP methods a.k.a. verbs.
enum HTTP_Method : uint8_t
  {
    http_method_null     = 0,
    http_method_get      = 1,
    http_method_head     = 2,
    http_method_post     = 3,
    http_method_put      = 4,
    http_method_delete   = 5,
    http_method_connect  = 6,
    http_method_options  = 7,
    http_method_trace    = 8,
  };

// Converts an HTTP method to a string such as `GET`.
// If the method is invalid, the invalid string `NULL` is returned.
ROCKET_CONST_FUNCTION
const char*
format_http_method(HTTP_Method method)
  noexcept;

// Parses a method from plain text.
// `http_method_null` is returned if the string is not valid.
ROCKET_PURE_FUNCTION
HTTP_Method
parse_http_method(const char* bptr, const char* eptr)
  noexcept;

// These are HTTP status codes.
// This list is not exhaustive. Custom values may be used.
enum HTTP_Status : uint16_t
  {
    http_status_null                    =   0,  // Null
    http_status_class_information       = 100,
    http_status_continue                = 100,  // Continue
    http_status_switching_protocol      = 101,  // Switching Protocol
    http_status_processing              = 102,  // Processing
    http_status_early_hints             = 103,  // Early Hints
    http_status_class_success           = 200,
    http_status_ok                      = 200,  // OK
    http_status_created                 = 201,  // Created
    http_status_accepted                = 202,  // Accepted
    http_status_nonauthoritative        = 203,  // Non-authoritative Information
    http_status_no_content              = 204,  // No Content
    http_status_reset_content           = 205,  // Reset Content
    http_status_partial_content         = 206,  // Partial Content
    http_status_multistatus             = 207,  // Multi-status
    http_status_already_reported        = 208,  // Already Reported
    http_status_im_used                 = 226,  // IM Used
    http_status_connection_established  = 299,  // Connection Established [x]
    http_status_class_redirection       = 300,
    http_status_multiple_choice         = 300,  // Multiple Choice
    http_status_moved_permanently       = 301,  // Moved Permanently
    http_status_found                   = 302,  // Found
    http_status_see_other               = 303,  // See Other
    http_status_not_modified            = 304,  // Not Modified
    http_status_use_proxy               = 305,  // Use Proxy
    http_status_temporary_redirect      = 307,  // Temporary Redirect
    http_status_permanent_redirect      = 308,  // Permanent Redirect
    http_status_class_client_error      = 400,
    http_status_bad_request             = 400,  // Bad Request
    http_status_unauthorized            = 401,  // Unauthorized
    http_status_forbidden               = 403,  // Forbidden
    http_status_not_found               = 404,  // Not Found
    http_status_method_not_allowed      = 405,  // Method Not Allowed
    http_status_not_acceptable          = 406,  // Not Acceptable
    http_status_proxy_unauthorized      = 407,  // Proxy Authentication Required
    http_status_request_timeout         = 408,  // Request Timeout
    http_status_conflict                = 409,  // Conflict
    http_status_gone                    = 410,  // Gone
    http_status_length_required         = 411,  // Length Required
    http_status_precondition_failed     = 412,  // Precondition Failed
    http_status_payload_too_large       = 413,  // Payload Too Large
    http_status_uri_too_long            = 414,  // URI Too Long
    http_status_unsupported_media_type  = 415,  // Unsupported Media Type
    http_status_range_not_satisfiable   = 416,  // Range Not Satisfiable
    http_status_expectation_failed      = 417,  // Expectation Failed
    http_status_misdirected_request     = 421,  // Misdirected Request
    http_status_unprocessable_entity    = 422,  // Unprocessable Entity
    http_status_locked                  = 423,  // Locked
    http_status_failed_dependency       = 424,  // Failed Dependency
    http_status_too_early               = 425,  // Too Early
    http_status_upgrade_required        = 426,  // Upgrade Required
    http_status_precondition_required   = 428,  // Precondition Required
    http_status_too_many_requests       = 429,  // Too Many Requests
    http_status_headers_too_large       = 431,  // Request Header Fields Too Large
    http_status_class_server_error      = 500,
    http_status_internal_server_error   = 500,  // Internal Server Error
    http_status_not_implemented         = 501,  // Not Implemented
    http_status_bad_gateway             = 502,  // Bad Gateway
    http_status_service_unavailable     = 503,  // Service Unavailable
    http_status_gateway_timeout         = 504,  // Gateway Timeout
    http_status_version_not_supported   = 505,  // HTTP Version Not Supported
    http_status_insufficient_storage    = 507,  // Insufficient Storage
    http_status_loop_detected           = 508,  // Loop Detected
    http_status_not_extended            = 510,  // Not Extended
    http_status_network_unauthorized    = 511,  // Network Authentication Required
  };

// Converts an HTTP status code to a string such as `Bad Request`.
// If the status code is unknown, `Unknown Status` is returned.
ROCKET_CONST_FUNCTION
const char*
describe_http_status(HTTP_Status stat)
  noexcept;

// Classifies a status code.
constexpr
HTTP_Status
classify_http_status(HTTP_Status stat)
  noexcept
  { return HTTP_Status(uint32_t(stat) / 100 * 100);  }

// These are internal states of HTTP and WebSocket encoders.
enum HTTP_Encoder_State : uint8_t
  {
    http_encoder_state_headers    = 0,
    http_encoder_state_closed     = 1,
    http_encoder_state_entity     = 2,
    http_encoder_state_tunnel     = 3,
    http_encoder_state_websocket  = 4,
  };

// These are internal states of HTTP and WebSocket decoders.
enum HTTP_Decoder_State : uint8_t
  {
  };

// These were designed for various options in the `Connection:` header, but
// now they denote [ method + connection + upgrade ] combinations.
enum HTTP_Connection : uint8_t
  {
    http_connection_keep_alive  = 0,
    http_connection_close       = 1,
    http_connection_upgrade     = 2,  // special value for pending upgrades
    http_connection_websocket   = 3,  // WebSocket
  };

// These are WebSocket opcodes.
// This list is exhaustive according to RFC 6455.
enum WebSocket_Opcode : uint8_t
  {
    websocket_opcode_continuation  = 0x00,
    websocket_opcode_text          = 0x10,
    websocket_opcode_binary        = 0x20,
    websocket_opcode_close         = 0x80,
    websocket_opcode_ping          = 0x90,
    websocket_opcode_pong          = 0xA0,
  };

// These are WebSocket status codes.
// This list is not exhaustive. Custom values may be used.
enum WebSocket_Status : uint16_t
  {
    websocket_status_null                =    0,
    websocket_status_normal_closure      = 1000,
    websocket_status_going_away          = 1001,
    websocket_status_protocol_error      = 1002,
    websocket_status_not_acceptable      = 1003,
    websocket_status_no_status           = 1005,  // reserved
    websocket_status_abnormal            = 1006,  // reserved
    websocket_status_data_error          = 1007,
    websocket_status_forbidden           = 1008,
    websocket_status_too_large           = 1009,
    websocket_status_extension_required  = 1010,  // reserved
    websocket_status_server_error        = 1011,
    websocket_status_tls_error           = 1015,  // reserved
  };

}  // namespace poseidon

#endif
