// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_CBPP_CONTROL_CODES_HPP_
#define POSEIDON_CBPP_CONTROL_CODES_HPP_

namespace Poseidon {

namespace Cbpp {
	typedef unsigned ControlCode;

	namespace ControlCodes {
		enum {						// vintParam		stringParam
			CTL_PING		=  0,	// 原样返回			原样返回
			CTL_SHUTDOWN	=  1,	// 0 	正常关闭	原样返回
									// 其它	暴力关闭	原样返回
		};
	}

	using namespace ControlCodes;
};

}

#endif
