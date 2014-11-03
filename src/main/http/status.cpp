#include "../precompiled.hpp"
#include "status.hpp"
#include <algorithm>
using namespace Poseidon;

namespace {

struct DescElement {
	unsigned status;
	const char *descShort;
	const char *descLong;
};

struct DescElementComparator {
	bool operator()(const DescElement &lhs, const DescElement &rhs) const {
		return lhs.status < rhs.status;
	}
	bool operator()(unsigned lhs, const DescElement &rhs) const {
		return lhs < rhs.status;
	}
	bool operator()(const DescElement &lhs, unsigned rhs) const {
		return lhs.status < rhs;
	}
};

const DescElement DESC_TABLE[] = {
	// http://www.w3.org/Protocols/rfc2616/rfc2616-sec10.html
	{ 100,	"Continue",
			"The client should continue with its request." },
	{ 101,	"Switching Protocols",
			"The server understands and is willing to comply with the client's "
			"request, via the Upgrade message header field, for a change in the "
			"application protocol being used on this connection." },
	{ 200,	"OK",
			"The request has succeeded." },
	{ 201,	"Created",
			"The request has been fulfilled and resulted in a new resource being "
			"created." },
	{ 202,	"Accepted",
			"The request has been accepted for processing, but the processing has "
			"not been completed." },
	{ 203,	"Non-Authoritative Information",
			"The returned metainformation in the entity-header is not the "
			"definitive set as available from the origin server, but is gathered "
			"from a local or a third-party copy." },
	{ 204,	"No Content",
			"The server has fulfilled the request but does not need to return an "
			"entity-body, and might want to return updated metainformation." },
	{ 205,	"Reset Content",
			"The server has fulfilled the request and the user agent should reset "
			"the document view which caused the request to be sent." },
	{ 206,	"Partial Content",
			"The server has fulfilled the partial GET request for the resource." },
	{ 300,	"Multiple Choices",
			"The requested resource corresponds to any one of a set of "
			"representations, each with its own specific location." },
	{ 301,	"Moved Permanently",
			"The requested resource has been assigned a new permanent URI and "
			"any future references to this resource should use one of the "
			"returned URIs." },
	{ 302,	"Found",
			"The requested resource resides temporarily under a different URI." },
	{ 303,	"See Other",
			"The response to the request can be found under a different URI and "
			"should be retrieved using a GET method on that resource." },
	{ 304,	"Not Modified",
			"The client has performed a conditional GET request and access is "
			"allowed, but the document has not been modified." },
	{ 305,	"Use Proxy",
			"The requested resource MUST be accessed through the proxy given by "
			"the Location field." },
	{ 307,	"Temporary Redirect",
			"The requested resource resides temporarily under a different URI." },
	{ 400,	"Bad Request",
			"The request could not be understood by the server due to malformed "
			"syntax." },
	{ 401,	"Unauthorized",
			"The request requires user authentication." },
	{ 403,	"Forbidden",
			"The server understood the request, but is refusing to fulfill it." },
	{ 404,	"Not Found",
			"The server has not found anything matching the Request-URI." },
	{ 405,	"Method Not Allowed",
			"The method specified in the Request-Line is not allowed for the "
			"resource identified by the Request-URI." },
	{ 406,	"Not Acceptable",
			"The resource identified by the request is only capable of "
			"generating response entities which have content characteristics not "
			"acceptable according to the accept headers sent in the request." },
	{ 407,	"Proxy Authentication Required",
			"The client must first authenticate itself with the proxy." },
	{ 408,	"Request Timeout",
			"The client did not produce a request within the time that the server "
			"was prepared to wait." },
	{ 409,	"Conflict",
			"The request could not be completed due to a conflict with the "
			"current state of the resource." },
	{ 410,	"Gone",
			"The requested resource is no longer available at the server "
			"and no forwarding address is known." },
	{ 411,	"Length Required",
			"The server refuses to accept the request without a defined "
			"Content-Length." },
	{ 412,	"Precondition Failed",
			"The precondition given in one or more of the request-header "
			"fields evaluated to false when it was tested on the server." },
	{ 413,	"Request Entity Too Large",
			"The server is refusing to process a request because the request "
			"entity is larger than the server is willing or able to process." },
	{ 414,	"Request-URI Too Long",
			"The server is refusing to service the request because the "
			"Request-URI is longer than the server is willing to interpret." },
	{ 415,	"Unsupported Media Type",
			"The server is refusing to service the request because the entity of "
			"the request is in a format not supported by the requested resource "
			"for the requested method." },
	{ 416,	"Requested Range Not Satisfiable",
			"The request included a Range request-header field, and none of the "
			"range-specifier values in this field overlap the current extent of "
			"the selected resource, and the request did not include an If-Range "
			"request-header field." },
	{ 417,	"Expectation Failed",
			"The expectation given in an Expect request-header field could not "
			"be met by this server." },
	{ 500,	"Internal Server Error",
			"The server encountered an unexpected condition which prevented it "
			"from fulfilling the request." },
	{ 501,	"Not Implemented",
			"The server does not support the functionality required to fulfill "
			"the request." },
	{ 502,	"Bad Gateway",
			"The server, while acting as a gateway or proxy, received an invalid "
			"response from the upstream server it accessed in attempting to "
			"fulfill the request." },
	{ 503,	"Service Unavailable",
			"The server is currently unable to handle the request due to a "
			"temporary overloading or maintenance of the server." },
	{ 504,	"Gateway Timeout",
			"The server, while acting as a gateway or proxy, did not receive a "
			"timely response from the upstream server specified by the URI "
			"(e.g. HTTP, FTP, LDAP) or some other auxiliary server (e.g. DNS) "
			"it needed to access in attempting to complete the request." },
	{ 505,	"HTTP Version Not Supported",
			"The server does not support, or refuses to support, the HTTP "
			"protocol version that was used in the request message." },
};

}

namespace Poseidon {

HttpStatusDesc getHttpStatusDesc(HttpStatus status){
	HttpStatusDesc ret;
	const AUTO(element,
		std::lower_bound(BEGIN(DESC_TABLE), END(DESC_TABLE),
			static_cast<unsigned>(status), DescElementComparator())
	);
	if((element != END(DESC_TABLE)) && (element->status == (unsigned)status)){
		ret.descShort = element->descShort;
		ret.descLong = element->descLong;
	} else {
		ret.descShort = "Unknown Status Code";
		ret.descLong = "No description available for this status code.";
	}
	return ret;
}

}
