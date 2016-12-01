// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2016, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_ZLIB_HPP_
#define POSEIDON_ZLIB_HPP_

#include "stream_buffer.hpp"

namespace Poseidon {

extern StreamBuffer deflate(StreamBuffer &src, int level = 6);
extern StreamBuffer inflate(StreamBuffer &src);

extern StreamBuffer gzip_deflate(StreamBuffer &src, int level = 6);
extern StreamBuffer gzip_inflate(StreamBuffer &src);

}

#endif
