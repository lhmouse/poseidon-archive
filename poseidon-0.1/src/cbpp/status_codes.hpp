// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2016, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_CBPP_STATUS_CODES_HPP_
#define POSEIDON_CBPP_STATUS_CODES_HPP_

namespace Poseidon {

namespace Cbpp {
	typedef int StatusCode;

	namespace StatusCodes {
		enum {
			ST_MONOTONIC_CLOCK      =    3,
			ST_SHUTDOWN_REQUEST     =    2,
			ST_PONG                 =    1,
			ST_OK                   =    0,

			ST_INTERNAL_ERROR       =   -1,
			ST_END_OF_STREAM        =   -2,
			ST_NOT_FOUND            =   -3,
			ST_REQUEST_TOO_LARGE    =   -4,
			ST_RESPONSE_TOO_LARGE   =   -5,
			ST_JUNK_AFTER_PACKET    =   -6,
			ST_FORBIDDEN            =   -7,
			ST_AUTH_REQUIRED        =   -8,
			ST_LENGTH_ERROR         =   -9,
			ST_UNKNOWN_CTL_CODE     =  -10,
		};
	}

	using namespace StatusCodes;
}

}

#endif
