// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_CBPP_STATUS_CODES_HPP_
#define POSEIDON_CBPP_STATUS_CODES_HPP_

namespace Poseidon {
namespace Cbpp {

using Status_code = long;

inline namespace Status_codes {
	enum {
		status_shutdown                =    3,
		status_pong                    =    2,
		status_ping                    =    1,
		status_ok                      =    0,

		status_internal_error          =   -1,
		status_end_of_stream           =   -2,
		status_not_found               =   -3,
		status_request_too_large       =   -4,
		status_bad_request             =   -5,
		status_junk_after_packet       =   -6,
		status_forbidden               =   -7,
		status_authorization_failure   =   -8,
		status_length_error            =   -9,
		status_unknown_control_code    =  -10,
		status_data_corrupted          =  -11,
		status_gone_away               =  -12,
		status_invalid_argument        =  -13,
		status_unsupported             =  -14,
	};
}

}
}

#endif
