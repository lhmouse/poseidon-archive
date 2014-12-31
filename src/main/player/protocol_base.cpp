// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "protocol_base.hpp"
#include "exception.hpp"
#include "status.hpp"
#include "../endian.hpp"
#include "../log.hpp"
using namespace Poseidon;

/*

协议说明：
每个数据包由头部和正文两部分组成。
其中头部按以下顺序的字段构成（规格均为小端序，每个字节 8 位）：

字段名	必需性	规格	说明
-------------------------------------------------------------------------------
小长度	必需	uint16	如果该字段的值不为 0xFFFF，则表示正文长度；否则，
						使用后面的大长度表示正文长度。
协议号	必需	uint16	供标识消息类型使用。

						唯一被保留的协议号是 0。0 号协议的格式为：
							uvint	辅助协议号
							uvint	状态码
							string	二进制消息

						服务端到客户端的 0 号协议为通用错误码返回：
							辅助协议号	指定该协议返回给客户端的哪个请求
							状态码		返回给该客户端请求的状态码
							二进制消息	返回给该客户端请求的消息

						客户端到服务端的 0 号协议用于控制会话状态：
							辅助协议号	指定要进行的操作：
										0	心跳包，用于保持连接。参数被忽略。
										其他值被保留。如果发送了被保留的值，
										服务端将返回同样的消息，然后关闭连接。
							状态码		作为参数使用。
							二进制消息	作为参数使用。

大长度	可选	uint64	仅在小长度为 0xFFFF 时使用。表示正文长度。

*/

void PlayerProtocolBase::encodeHeader(
	StreamBuffer &dst, boost::uint16_t protocolId, boost::uint64_t protocolLen)
{
	boost::uint16_t temp16;

	// 小长度，必需。
	storeLe(temp16, (protocolLen < 0xFFFF) ? protocolLen : 0xFFFF);
	dst.put(&temp16, 2);

	// 协议号，必需。
	storeLe(temp16, protocolId);
	dst.put(&temp16, 2);

	if(protocolLen >= 0xFFFF){
		// 大长度，可选。
		boost::uint64_t temp64;
		storeLe(temp64, protocolLen);
		dst.put(&temp64, 8);
	}
}
bool PlayerProtocolBase::decodeHeader( // 如果返回 false，无事发生。
	boost::uint16_t &protocolId, boost::uint64_t &protocolLen, StreamBuffer &src) NOEXCEPT
{
	boost::uint16_t temp16;
	if(src.peek(&temp16, 2) < 2){
		return false;
	}
	// （小长度 + 协议号）一共 4 字节。
	std::size_t totalLen = 4;
	// 小长度，必需。
	protocolLen = loadLe(temp16);
	if(protocolLen == 0xFFFF){
		totalLen += 8;
	}
	if(src.size() < totalLen){
		return false;
	}
	src.discard(2);

	// 协议号，必需。
	src.get(&temp16, 2);
	protocolId = loadLe(temp16);

	if(protocolLen == 0xFFFF){
		// 大长度，可选。
		boost::uint64_t temp64;
		src.get(&temp64, 8);
		protocolLen = loadLe(temp64);
	}
	return true;
}
