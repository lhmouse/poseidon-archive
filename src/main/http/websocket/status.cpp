#include "../../../precompiled.hpp"
#include "status.hpp"
#include <algorithm>
using namespace Poseidon;

namespace {

struct DescElement {
	unsigned status;
	const char *desc;
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
	{ 1000, "Normal Closure" },
	{ 1001, "Going Away" },
	{ 1002, "Protocol Error" },
	{ 1003, "Inacceptable Payload" },
	{ 1004, "Reserved Unknown" },
	{ 1005, "Reserved No Status" },
	{ 1006, "Reserved Abnormal" },
	{ 1007, "Inconsistent Payload" },
	{ 1008, "Access Denied" },
	{ 1009, "Message Too Big" },
	{ 1010, "Extension Not Available" },
	{ 1011, "Internal Server Error" },
	{ 1015, "Reserved TLS Error" }
};

}

namespace Poseidon {

const char *getWebSocketStatusDesc(WebSocketStatus status){
	const AUTO(element,
		std::lower_bound(BEGIN(DESC_TABLE), END(DESC_TABLE),
			static_cast<unsigned>(status), DescElementComparator())
	);
	if((element != END(DESC_TABLE)) && (element->status == (unsigned)status)){
		return element->desc;
	}
	return "Unknown Status Code";
}

}
