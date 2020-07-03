// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "enums.hpp"
#include "../utilities.hpp"

namespace poseidon {

const char*
describe_http_version(HTTP_Version ver)
noexcept
  {
    switch(ver) {
      case http_version_1_0:
        return "HTTP/1.0";

      case http_version_1_1:
        return "HTTP/1.1";

      case http_version_null:
      default:
        return "HTTP/0.0";
    }
  }

HTTP_Version
parse_http_version(const char* bptr, const char* eptr)
  {
    // Check hard-coded characters.
    if((eptr - bptr != 8) || (::std::memcmp(bptr, "HTTP/", 5) != 0) || (bptr[6] != '.'))
      POSEIDON_THROW("Invalid HTTP version string");

    // Parse version numbers.
    uint32_t maj = static_cast<uint8_t>(bptr[5]);
    uint32_t min = static_cast<uint8_t>(bptr[7]);

    if((maj < '1') || ('9' < maj) || (min < '0') || ('9' < min))
      POSEIDON_THROW("Invalid HTTP version `$1.$2`", bptr[5], bptr[7]);

    maj -= '0';
    min -= '0';

    // Build the version number.
    return static_cast<HTTP_Version>(maj << 8 | min);
  }

const char*
describe_http_status(HTTP_Status stat)
noexcept
  {
    switch(stat) {
      case http_status_continue:
        return "Continue";

      case http_status_switching_protocol:
        return "Switching Protocol";

      case http_status_processing:
        return "Processing";

      case http_status_early_hints:
        return "Early Hints";

      case http_status_ok:
        return "OK";

      case http_status_created:
        return "Created";

      case http_status_accepted:
        return "Accepted";

      case http_status_nonauthoritative:
        return "Non-authoritative Information";

      case http_status_no_content:
        return "No Content";

      case http_status_reset_content:
        return "Reset Content";

      case http_status_partial_content:
        return "Partial Content";

      case http_status_multistatus:
        return "Multi-status";

      case http_status_already_reported:
        return "Already Reported";

      case http_status_im_used:
        return "IM Used";

      case http_status_multiple_choice:
        return "Multiple Choice";

      case http_status_moved_permanently:
        return "Moved Permanently";

      case http_status_found:
        return "Found";

      case http_status_see_other:
        return "See Other";

      case http_status_not_modified:
        return "Not Modified";

      case http_status_use_proxy:
        return "Use Proxy";

      case http_status_temporary_redirect:
        return "Temporary Redirect";

      case http_status_permanent_redirect:
        return "Permanent Redirect";

      case http_status_bad_request:
        return "Bad Request";

      case http_status_unauthorized:
        return "Unauthorized";

      case http_status_forbidden:
        return "Forbidden";

      case http_status_not_found:
        return "Not Found";

      case http_status_method_not_allowed:
        return "Method Not Allowed";

      case http_status_not_acceptable:
        return "Not Acceptable";

      case http_status_proxy_unauthorized:
        return "Proxy Authentication Required";

      case http_status_request_timedout:
        return "Request Timeout";

      case http_status_conflict:
        return "Conflict";

      case http_status_gone:
        return "Gone";

      case http_status_length_required:
        return "Length Required";

      case http_status_precondition_failed:
        return "Precondition Failed";

      case http_status_payload_too_large:
        return "Payload Too Large";

      case http_status_uri_too_long:
        return "URI Too Long";

      case http_status_unsupported_media:
        return "Unsupported Media Type";

      case http_status_range_not_satisfiable:
        return "Range Not Satisfiable";

      case http_status_expectation_failed:
        return "Expectation Failed";

      case http_status_misdirected_request:
        return "Misdirected Request";

      case http_status_unprocessable:
        return "Unprocessable Entity";

      case http_status_locked:
        return "Locked";

      case http_status_failed_dependency:
        return "Failed Dependency";

      case http_status_too_early:
        return "Too Early";

      case http_status_upgrade_required:
        return "Upgrade Required";

      case http_status_precondition_required:
        return "Precondition Required";

      case http_status_too_many_requests:
        return "Too Many Requests";

      case http_status_headers_too_large:
        return "Request Header Fields Too Large";

      case http_status_internal_server_error:
        return "Internal Server Error";

      case http_status_not_implemented:
        return "Not Implemented";

      case http_status_bad_gateway:
        return "Bad Gateway";

      case http_status_service_unavailable:
        return "Service Unavailable";

      case http_status_gateway_timeout:
        return "Gateway Timeout";

      case http_status_version_not_supported:
        return "HTTP Version Not Supported";

      case http_status_insufficient_storage:
        return "Insufficient Storage";

      case http_status_loop_detected:
        return "Loop Detected";

      case http_status_not_extended:
        return "Not Extended";

      case http_status_network_unauthorized:
        return "Network Authentication Required";

      case http_status_null:
      default:
        return "Unknown Status";
    }
  }

}  // namespace poseidon