// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_WEBSOCKET_STATUS_CODES_HPP_
#define POSEIDON_WEBSOCKET_STATUS_CODES_HPP_

namespace Poseidon {
namespace Websocket {

using Status_code = int;

inline namespace Status_codes {
	enum {
		status_invalid              =   -1,
		status_normal_closure       = 1000,
		status_going_away           = 1001,
		status_protocol_error       = 1002,
		status_inacceptable         = 1003,
		status_reserved_unknown     = 1004,
		status_reserved_no_status   = 1005,
		status_reserved_abnormal    = 1006,
		status_inconsistent         = 1007,
		status_access_denied        = 1008,
		status_message_too_large    = 1009,
		status_extension_not_avail  = 1010,
		status_internal_error       = 1011,
		status_reserved_tls         = 1015,
	};
}

}
}

#endif
