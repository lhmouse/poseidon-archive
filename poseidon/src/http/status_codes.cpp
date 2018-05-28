// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "status_codes.hpp"
#include "../cxx_ver.hpp"
#include <algorithm>

namespace Poseidon {
namespace Http {

namespace {
	struct Status_desc_element {
		Status_code status_code;
		Status_code_desc desc;
	};

	inline bool operator<(const Status_desc_element &elem, Status_code status_code){
		return elem.status_code < status_code;
	}
	inline bool operator<(Status_code status_code, const Status_desc_element &elem){
		return status_code < elem.status_code;
	}

	constexpr Status_desc_element s_desc_table[] = {
		// https://www.rfc-editor.org/rfc/rfc7231.txt
		{ 100, { "Continue",
		         "The initial part of a request has been received and has not yet been rejected by the server." } },
		{ 101, { "Switching Protocols",
		         "The server understands and is willing to comply with the client's request, via the Upgrade header field, "
		         "for a change in the application protocol being used on this connection." } },
		{ 200, { "OK",
		         "The client's request was successfully received, understood, and accepted." } },
		{ 201, { "Created",
		         "The request has been fulfilled and has resulted in one or more new resources being created." } },
		{ 202, { "Accepted",
		         "The request has been accepted for processing, but the processing has not been completed." } },
		{ 203, { "Non-Authoritative Information",
		         "The request was successful but the enclosed payload has been modified from that of the origin server's "
		         "200 (OK) response by a transforming proxy." } },
		{ 204, { "No Content",
		         "The server has successfully fulfilled the request and there is no additional content to send "
		         "in the response payload body." } },
		{ 205, { "Reset Content",
		         "The server has fulfilled the request and desires that the user agent reset the \"document view\", "
		         "which caused the request to be sent, to its original state as received from the origin server." } },
		{ 206, { "Partial Content",
		         "The server is successfully fulfilling a range request for the target resource by transferring "
		         "one or more parts of the selected representation that correspond to the satisfiable ranges found "
		         "in the request's Range header field." } },
		{ 300, { "Multiple Choices",
		         "The target resource has more than one representation, each with its own more specific identifier, "
		         "and information about the alternatives is being provided so that the user (or user agent) "
		         "can select a preferred representation by redirecting its request to one or more of those identifiers." } },
		{ 301, { "Moved Permanently",
		         "The target resource has been assigned a new permanent URI and any future references to this resource "
		         "ought to use one of the enclosed URIs." } },
		{ 302, { "Found",
		         "The target resource resides temporarily under a different URI." } },
		{ 303, { "See Other",
		         "The server is redirecting the user agent to a different resource, as indicated by a URI "
		         "in the Location header field, which is intended to provide an indirect response to the original request." } },
		{ 304, { "Not Modified",
		         "A conditional GET or HEAD request has been received and would have resulted in a 200 (OK) response "
		         "if it were not for the fact that the condition evaluated to false." } },
		{ 305, { "Use Proxy",
		         "The requested resource MUST be accessed through the proxy given by the Location field." } },
		{ 307, { "Temporary Redirect",
		         "The target resource resides temporarily under a different URI and the user agent MUST NOT change "
		         "the request method if it performs an automatic redirection to that URI." } },
		{ 400, { "Bad Request",
		         "The server cannot or will not process the request due to something that is perceived to be a client error "
		         "(e.g., malformed request syntax, invalid request message framing, or deceptive request routing)." } },
		{ 401, { "Unauthorized",
		         "The request has not been applied because it lacks valid authentication credentials for the target resource." } },
		{ 403, { "Forbidden",
		         "The server understood the request but refuses to authorize it." } },
		{ 404, { "Not Found",
		         "The origin server did not find a current representation for the target resource or is not willing "
		         "to disclose that one exists." } },
		{ 405, { "Method Not Allowed",
		         "The method received in the request-line is known by the origin server but not supported by the target resource." } },
		{ 406, { "Not Acceptable",
		         "The target resource does not have a current representation that would be acceptable to the user agent, "
		         "according to the proactive negotiation header fields received in the request, and the server is unwilling "
		         "to supply a default representation." } },
		{ 407, { "Proxy Authentication Required",
		         "The client needs to authenticate itself in order to use a proxy." } },
		{ 408, { "Request Timeout",
		         "The server did not receive a complete request message within the time that it was prepared to wait." } },
		{ 409, { "Conflict",
		         "The request could not be completed due to a conflict with the current state of the target resource." } },
		{ 410, { "Gone",
		         "Access to the target resource is no longer available at the origin server and that this condition "
		         "is likely to be permanent." } },
		{ 411, { "Length Required",
		         "The server refuses to accept the request without a defined Content-Length." } },
		{ 412, { "Precondition Failed",
		         "One or more conditions given in the request header fields evaluated to false when tested on the server." } },
		{ 413, { "Payload Too Large",
		         "The server is refusing to process a request because the request payload is larger than the server "
		         "is willing or able to process." } },
		{ 414, { "URI Too Long",
		         "The server is refusing to service the request because the request-target is longer than the server "
		         "is willing to interpret." } },
		{ 415, { "Unsupported Media Type",
		         "The origin server is refusing to service the request because the payload is in a format not supported "
		         "by this method on the target resource." } },
		{ 416, { "Range Not Satisfiable",
		         "None of the ranges in the request's Range header field overlap the current extent of the selected resource "
		         "or the set of ranges requested has been rejected due to invalid ranges or an excessive request "
		         "of small or overlapping ranges." } },
		{ 417, { "Expectation Failed",
		         "The expectation given in the request's Expect header field could not be met by at least one of the inbound servers." } },
		{ 426, { "Upgrade Required",
		         "The server refuses to perform the request using the current protocol but might be willing to do so "
		         "after the client upgrades to a different protocol." } },
		{ 500, { "Internal Server Error",
		         "The server encountered an unexpected condition that prevented it from fulfilling the request." } },
		{ 501, { "Not Implemented",
		         "The server does not support the functionality required to fulfill the request." } },
		{ 502, { "Bad Gateway",
		         "The server, while acting as a gateway or proxy, received an invalid response from an inbound server it accessed "
		         "while attempting to fulfill the request." } },
		{ 503, { "Service Unavailable",
		         "The server is currently unable to handle the request due to a temporary overload or scheduled maintenance, "
		         "which will likely be alleviated after some delay." } },
		{ 504, { "Gateway Timeout",
		         "The server, while acting as a gateway or proxy, did not receive a timely response from an upstream server "
		         "it needed to access in order to complete the request." } },
		{ 505, { "HTTP Version Not Supported",
		         "The server does not support, or refuses to support, the major version of  HTTP that was used in the request message." } },
		{ 999, { "Unknown Status Code",
		         "No description available for this status code." } },
	};
}

Status_code_desc get_status_code_desc(Status_code status_code) noexcept {
	const auto begin = std::begin(s_desc_table);
	const auto end = std::end(s_desc_table) - 1; // This points to the last element for invalid status codes.
	const auto ptr = std::lower_bound(begin, end, status_code);
	if(ptr->status_code != status_code){
		return end->desc;
	}
	return ptr->desc;
}

}
}
