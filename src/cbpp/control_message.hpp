// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_CBPP_CONTROL_MESSAGE_HPP_
#define POSEIDON_CBPP_CONTROL_MESSAGE_HPP_

#include "message_base.hpp"

namespace Poseidon {

namespace Cbpp {

#define MESSAGE_NAME	ControlMessage
#define MESSAGE_ID		0
#define MESSAGE_FIELDS	\
	FIELD_VUINT			(controlCode)	\
	FIELD_VINT			(vintParam)	\
	FIELD_STRING		(stringParam)
#include "message_generator.hpp"

}

}

#endif
