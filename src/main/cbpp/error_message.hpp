// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_CBPP_ERROR_MESSAGE_HPP_
#define POSEIDON_CBPP_ERROR_MESSAGE_HPP_

#include "message_base.hpp"

namespace Poseidon {

#define MESSAGE_NAME	CbppErrorMessage
#define MESSAGE_ID		0
#define MESSAGE_FIELDS	\
	FIELD_VUINT			(messageId)	\
	FIELD_VINT			(status)	\
	FIELD_STRING		(reason)
#include "message_generator.hpp"

}

#endif
