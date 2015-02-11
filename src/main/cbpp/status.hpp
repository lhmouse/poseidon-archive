// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_CBPP_STATUS_HPP_
#define POSEIDON_CBPP_STATUS_HPP_

namespace Poseidon {

namespace CbppStatusCodes {
	typedef long long CbppStatus;

	enum {
		CBPP_OK					= 0,
		CBPP_INTERNAL_ERROR		= -1,
		CBPP_END_OF_STREAM		= -2,
		CBPP_NOT_FOUND			= -3,
		CBPP_REQUEST_TOO_LARGE	= -4,
		CBPP_RESPONSE_TOO_LARGE	= -5,
		CBPP_JUNK_AFTER_PACKET	= -6,
	};
}

using namespace CbppStatusCodes;

}

#endif
