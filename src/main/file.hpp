#ifndef POSEIDON_FILE_HPP_
#define POSEIDON_FILE_HPP_

#include <string>
#include "shared_ntmbs.hpp"
#include "stream_buffer.hpp"

namespace Poseidon {

extern void fileGetContents(StreamBuffer &contents, const SharedNtmbs &path);
extern int fileGetContentsNoThrow(StreamBuffer &contents, const SharedNtmbs &path);

extern void filePutContents(const SharedNtmbs &path, StreamBuffer contents, bool append);
extern int filePutContentsNoThrow(const SharedNtmbs &path, StreamBuffer contents, bool append);

extern bool getLine(StreamBuffer &buffer, std::string &line);

}

#endif
