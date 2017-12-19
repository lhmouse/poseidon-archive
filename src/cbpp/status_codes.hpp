// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2017, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_CBPP_STATUS_CODES_HPP_
#define POSEIDON_CBPP_STATUS_CODES_HPP_

namespace Poseidon {
namespace Cbpp {

typedef int StatusCode;

namespace StatusCodes {
	enum {
		ST_SHUTDOWN                =    3,
		ST_PONG                    =    2,
		ST_PING                    =    1,
		ST_OK                      =    0,

		ST_INTERNAL_ERROR          =   -1,
		ST_END_OF_STREAM           =   -2,
		ST_NOT_FOUND               =   -3,
		ST_REQUEST_TOO_LARGE       =   -4,
		ST_BAD_REQUEST             =   -5,
		ST_JUNK_AFTER_PACKET       =   -6,
		ST_FORBIDDEN               =   -7,
		ST_AUTHORIZATION_FAILURE   =   -8,
		ST_LENGTH_ERROR            =   -9,
		ST_UNKNOWN_CONTROL_CODE    =  -10,
		ST_DATA_CORRUPTED          =  -11,
		ST_GONE_AWAY               =  -12,
		ST_INVALID_ARGUMENT        =  -13,
		ST_UNSUPPORTED             =  -14,
	};
}

using namespace StatusCodes;

}
}

#endif
