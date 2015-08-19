// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_FILE_HPP_
#define POSEIDON_FILE_HPP_

#include <string>
#include "shared_nts.hpp"
#include "stream_buffer.hpp"

namespace Poseidon {

extern void fileGetContents(StreamBuffer &contents, const char *path);
extern int fileGetContentsNoThrow(StreamBuffer &contents, const char *path);

extern void filePutContents(const char *path, StreamBuffer contents, bool append);
extern int filePutContentsNoThrow(const char *path, StreamBuffer contents, bool append);

}

#endif
