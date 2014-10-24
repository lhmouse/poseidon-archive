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
	{ 100,	"Continue",
			"The request can be continued." },
	{ 101,	"Switch Protocols",
			"The server has switched protocols in an upgrade header." },
	{ 200,	"OK",
			"The request completed successfully." },
	{ 201,	"Created",
			"The request has been fulfilled and resulted in the creation "
			"of a new resource." },
	{ 202,	"Accepted",
			"The request has been accepted for processing, but "
			"the processing has not been completed." },
	{ 203,	"Partial",
			"The returned meta information in the entity-header is not "
			"the definitive set available from the originating server." },
	{ 204,	"No Content",
			"The server has fulfilled the request, but there is no "
			"new information to send back." },
	{ 205,	"Reset Content",
			"The request has been completed, and the client program should "
			"reset the document view that caused the request to be sent to "
			"allow the user to easily initiate another "
			"input action." },
	{ 206,	"Partial Content",
			"The server has fulfilled the partial GET request for the resource." },
	{ 207,	"WebDAV Multi Status",
			"During a World Wide Web Distributed Authoring and Versioning "
			"(WebDAV) operation, this indicates multiple status codes for "
			"a single response. The response body contains Extensible Markup "
			"Language (XML) that describes the status codes. For more "
			"information, see HTTP Extensions for Distributed Authoring." },
	{ 300,	"Ambiguous",
			"The requested resource is available at one or more locations." },
	{ 301,	"Moved",
			"The requested resource has been assigned to a new permanent "
			"Uniform Resource Identifier (URI), and any future references "
			"to this resource should be done using one of the returned URIs." },
	{ 302,	"Redirect",
			"The requested resource resides temporarily under a different URI." },
	{ 303,	"Redirect Method",
			"The response to the request can be found under a different URI and "
			"should be retrieved using a GET HTTP verb on that resource." },
	{ 304,	"Not Modified",
			"The requested resource has not been modified." },
	{ 305,	"Use Proxy",
			"The requested resource must be accessed through the proxy given by "
			"the location field." },
	{ 307,	"Redirect Keep Verb",
			"The redirected request keeps the same HTTP verb. HTTP/1.1 behavior." },
	{ 400,	"Bad Request",
			"The request could not be processed by the server due to invalid syntax." },
	{ 401,	"Denied",
			"The requested resource requires user authentication." },
	{ 402,	"Payment Req",
			"Not implemented in the HTTP protocol." },
	{ 403,	"Forbidden",
			"The server understood the request, but cannot fulfill it." },
	{ 404,	"Not Found",
			"The server has not found anything that matches the requested URI." },
	{ 405,	"Bad Method",
			"The HTTP verb used is not allowed." },
	{ 406,	"None Acceptable",
			"No responses acceptable to the client were found." },
	{ 407,	"Proxy Auth Req",
			"Proxy authentication required." },
	{ 408,	"Request Timeout",
			"The server timed out waiting for the request." },
	{ 409,	"Conflict",
			"The request could not be completed due to a conflict with the current "
			"state of the resource. The user should resubmit with more information." },
	{ 410,	"Gone",
			"The requested resource is no longer available at the server, and no "
			"forwarding address is known." },
	{ 411,	"Length Required",
			"The server cannot accept the request without a defined content length." },
	{ 412,	"Precond Failed",
			"The precondition given in one or more of the request header fields "
			"evaluated to false when it was tested on the server." },
	{ 413,	"Request Too Large",
			"The server cannot process the request because the request entity is larger "
			"than the server is able to process." },
	{ 414,	"URI Too Long",
			"The server cannot service the request because the request URI is longer "
			"than the server can interpret." },
	{ 415,	"Unsupported Media",
			"The server cannot service the request because the entity of the request "
			"is in a format not supported by the requested resource for the requested "
			"method." },
	{ 449,	"Retry With",
			"The request should be retried after doing the appropriate action." },
	{ 500,	"Server Error",
			"The server encountered an unexpected condition that prevented it from "
			"fulfilling the request." },
	{ 501,	"Not Supported",
			"The server does not support the functionality required to fulfill "
			"the request." },
	{ 502,	"Bad Gateway",
			"The server, while acting as a gateway or proxy, received an invalid "
			"response from the upstream server it accessed in attempting to fulfill "
			"the request." },
	{ 503,	"Service Unavail",
			"The service is temporarily overloaded." },
	{ 504,	"Gateway Timeout",
			"The request was timed out waiting for a gateway." },
	{ 505,	"Version Not Sup",
			"The server does not support the HTTP protocol version that was used "
			"in the request message." }
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
