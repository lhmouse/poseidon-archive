// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_CBPP_MESSAGE_BASE_HPP_
#define POSEIDON_CBPP_MESSAGE_BASE_HPP_

#include "../cxx_ver.hpp"
#include "../cxx_util.hpp"
#include <string>
#include <ostream>
#include <cstddef>
#include <boost/array.hpp>
#include <boost/container/vector.hpp>
#include <boost/container/deque.hpp>
#include <boost/cstdint.hpp>
#include "../vint64.hpp"
#include "../stream_buffer.hpp"
#include "../hex_printer.hpp"
#include "exception.hpp"
#include "status_codes.hpp"

/*===========================================================================*\

                             ---=* CBPP 说明 *=---

每个数据包由头部和正文两部分组成。
其中头部按以下顺序的字段构成（规格均为大端序，每个字节 8 位）：

字段名    必需性  规格      说明
-------------------------------------------------------------------------------
小长度    必需    uint16    如果该字段的值不为 0xFFFF，则表示正文长度；
                            否则，使用后面的大长度表示正文长度。
大长度    可选    uint64    仅在小长度为 0xFFFF 时使用。表示正文长度。
消息号    必需    uint16    供标识消息类型使用。
-------------------------------------------------------------------------------

消息号为零的消息称为控制消息。服务端发给客户端的控制消息也称为状态消息。
控制消息的格式为：

字段名    必需性  规格      说明
-------------------------------------------------------------------------------
状态码    必需    int32     用于控制链接状态。如果发送端发送了一个状态码
                            小于零的控制消息，说明其遇到了无法恢复的错误，
                            接收端必须立即关闭链接 。
参数      可选    flexible  这个字段的意义随控制码不同而不同。
-------------------------------------------------------------------------------

\*===========================================================================*/

#define THROW_END_OF_STREAM_(message_, field_)	\
	DEBUG_THROW(::Poseidon::Cbpp::Exception,	\
		::Poseidon::Cbpp::status_end_of_stream, ::Poseidon::Rcnts::view(	\
			"End of stream encountered in message " TOKEN_TO_STR(message_) " while parsing " TOKEN_TO_STR(field_) ))

#define THROW_JUNK_AFTER_PACKET_(message_)	\
	DEBUG_THROW(::Poseidon::Cbpp::Exception,	\
		::Poseidon::Cbpp::status_junk_after_packet, ::Poseidon::Rcnts::view(	\
			"Junk after message " TOKEN_TO_STR(message_) ))

#define THROW_LENGTH_ERROR_(message_, field_)	\
	DEBUG_THROW(::Poseidon::Cbpp::Exception,	\
		::Poseidon::Cbpp::status_length_error, ::Poseidon::Rcnts::view(	\
			"Length error in message " TOKEN_TO_STR(message_) " while parsing " TOKEN_TO_STR(field_) ))

namespace Poseidon {
namespace Cbpp {

class Message_base {
public:
	virtual ~Message_base();

public:
	virtual boost::uint64_t get_id() const = 0;
	virtual void serialize(Stream_buffer &buffer) const = 0;
	virtual void deserialize(Stream_buffer &buffer) = 0;
	virtual void dump_debug(std::ostream &os) const = 0;

public:
	operator Stream_buffer() const {
		Stream_buffer buffer;
		serialize(buffer);
		return buffer;
	}
};

extern std::ostream &operator<<(std::ostream &os, const Message_base &rhs);

}
}

#endif
