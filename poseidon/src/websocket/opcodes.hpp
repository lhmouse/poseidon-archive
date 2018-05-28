// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_WEBSOCKET_OPCODES_HPP_
#define POSEIDON_WEBSOCKET_OPCODES_HPP_

namespace Poseidon {
namespace Websocket {

using Opcode = int;

inline namespace Opcodes {
	enum {
		opmask_fin           = 0x80,
		opmask_rsv1          = 0x40,
		opmask_rsv2          = 0x20,
		opmask_rsv3          = 0x10,

		opmask_control       = 0x08,
		opmask_opcode        = 0x0F,

		opcode_invalid       =   -1,
		opcode_continuation  = 0x00,
		opcode_data_text     = 0x01,
		opcode_data_binary   = 0x02,
		opcode_close         = 0x08,
		opcode_ping          = 0x09,
		opcode_pong          = 0x0A,
	};
}

}
}

#endif
