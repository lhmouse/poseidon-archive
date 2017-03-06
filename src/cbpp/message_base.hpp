// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2017, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_CBPP_MESSAGE_BASE_HPP_
#define POSEIDON_CBPP_MESSAGE_BASE_HPP_

#include "../cxx_ver.hpp"
#include "../cxx_util.hpp"
#include <string>
#include <vector>
#include <algorithm>
#include <iomanip>
#include <ostream>
#include <cstddef>
#include <boost/array.hpp>
#include <boost/cstdint.hpp>
#include "../vint64.hpp"
#include "../stream_buffer.hpp"
#include "exception.hpp"
#include "status_codes.hpp"

/*===========================================================================*\
--------------------------------=* CBPP 说明 *=--------------------------------

每个数据包由头部和正文两部分组成。
其中头部按以下顺序的字段构成（规格均为小端序，每个字节 8 位）：

字段名  必需性  规格    说明
-------------------------------------------------------------------------------
小长度  必需    uint16  如果该字段的值不为 0xFFFF，则表示正文长度；否则，
                        使用后面的大长度表示正文长度。
协议号  必需    uint16  供标识消息类型使用。
大长度  可选    uint64  仅在小长度为 0xFFFF 时使用。表示正文长度。
-------------------------------------------------------------------------------

唯一被保留的协议号是 0。0 号协议的格式为：
  vint          状态码
  string        状态描述

服务端到客户端作为请求回执：
  状态码        返回该请求的状态码
  状态描述      返回该请求的状态描述

客户端到服务端用于控制会话状态：
  状态码        指定要进行的操作
                0   Ping 请求。状态描述参数被原样返回。
                1   关闭连接请求。
                （其他值被保留，不应当被使用。）
  状态描述      作为参数使用
\*===========================================================================*/

#define THROW_END_OF_STREAM_(message_, field_)	\
	DEBUG_THROW(::Poseidon::Cbpp::Exception,	\
		::Poseidon::Cbpp::ST_END_OF_STREAM, ::Poseidon::sslit(	\
			"End of stream encountered, expecting " TOKEN_TO_STR(message_) "::" TOKEN_TO_STR(field_) ))

#define THROW_JUNK_AFTER_PACKET_(message_)	\
	DEBUG_THROW(::Poseidon::Cbpp::Exception,	\
		::Poseidon::Cbpp::ST_JUNK_AFTER_PACKET, ::Poseidon::sslit(	\
			"Junk after message " TOKEN_TO_STR(message_) ))

#define THROW_LENGTH_ERROR_(message_, field_)	\
	DEBUG_THROW(::Poseidon::Cbpp::Exception,	\
		::Poseidon::Cbpp::ST_LENGTH_ERROR, ::Poseidon::sslit(	\
			"Length error in message field " TOKEN_TO_STR(message_) "::" TOKEN_TO_STR(field_) ))

namespace Poseidon {

namespace Cbpp {
	class MessageBase {
	public:
		virtual ~MessageBase();

	public:
		virtual unsigned get_message_id() const = 0;
		virtual void serialize(StreamBuffer &buffer) const = 0;
		virtual void deserialize(StreamBuffer &buffer) = 0;
		virtual void dump_debug(std::ostream &os) const = 0;

	public:
		operator StreamBuffer() const {
			StreamBuffer buffer;
			serialize(buffer);
			return buffer;
		}
	};

	extern std::ostream &operator<<(std::ostream &os, const MessageBase &rhs);
}

}

#endif
