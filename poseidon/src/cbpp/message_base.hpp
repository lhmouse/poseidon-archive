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
#include "../stream_buffer.hpp"

namespace Poseidon {
namespace Cbpp {

class Message_base {
public:
	virtual ~Message_base();

public:
	virtual boost::uint64_t get_id() const = 0;
	virtual void serialize(Stream_buffer &buffer) const = 0;
	virtual void deserialize(Stream_buffer &buffer) = 0;
	virtual void dump_debug(std::ostream &os, int indent_initial = 0) const = 0;

public:
	POSEIDON_ENABLE_IF_CXX11(explicit) operator Stream_buffer() const {
		Stream_buffer buffer;
		serialize(buffer);
		return buffer;
	}
};

inline std::ostream & operator<<(std::ostream &os, const Message_base &rhs){
	rhs.dump_debug(os);
	return os;
}

extern void shift_vint(boost::int64_t &value, Stream_buffer &buf, const char *name);
extern void shift_vuint(boost::uint64_t &value, Stream_buffer &buf, const char *name);
extern void shift_string(std::string &value, Stream_buffer &buf, const char *name);
extern void shift_blob(Stream_buffer &value, Stream_buffer &buf, const char *name);
extern void shift_fixed(void *data, std::size_t size, Stream_buffer &buf, const char *name);
extern void shift_flexible(Stream_buffer &value, Stream_buffer &buf, const char *name);

extern void push_vint(Stream_buffer &buf, boost::int64_t value);
extern void push_vuint(Stream_buffer &buf, boost::uint64_t value);
extern void push_string(Stream_buffer &buf, const std::string &value);
extern void push_blob(Stream_buffer &buf, const Stream_buffer &value);
extern void push_fixed(Stream_buffer &buf, const void *data, std::size_t size);
extern void push_flexible(Stream_buffer &buf, const Stream_buffer &value);

}
}

#endif
