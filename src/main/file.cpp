// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014, LH_Mouse. All wrongs reserved.

#include "precompiled.hpp"
#include "file.hpp"
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include "raii.hpp"
#include "exception.hpp"
using namespace Poseidon;

namespace Poseidon {

void fileGetContents(StreamBuffer &contents, const char *path){
	const UniqueFile file(::open(path, O_RDONLY));
	if(!file){
		DEBUG_THROW(SystemError);
	}
	StreamBuffer temp;
	for(;;){
		char readBuf[4096];
		const ::ssize_t bytesRead = ::read(file.get(), readBuf, sizeof(readBuf));
		if(bytesRead < 0){
			DEBUG_THROW(SystemError);
		} else if(bytesRead == 0){
			break;
		}
		temp.put(readBuf, bytesRead);
	}
	contents.swap(temp);
}
int fileGetContentsNoThrow(StreamBuffer &contents, const char *path){
	try {
		fileGetContents(contents, path);
		return 0;
	} catch(SystemError &e){
		return e.code();
	}
}

void filePutContents(const char *path, StreamBuffer contents, bool append){
	const UniqueFile file(::open(path, O_WRONLY | O_CREAT | (append ? O_APPEND : O_TRUNC)));
	if(!file){
		DEBUG_THROW(SystemError);
	}
	for(;;){
		char writeBuf[4096];
		const ::ssize_t bytesToWrite = contents.get(writeBuf, sizeof(writeBuf));
		if(bytesToWrite == 0){
			break;
		}
		if(::write(file.get(), writeBuf, bytesToWrite) < bytesToWrite){
			DEBUG_THROW(SystemError);
		}
	}
}
int filePutContentsNoThrow(const char *path, StreamBuffer contents, bool append){
	try {
		filePutContents(path, contents, append);
		return 0;
	} catch(SystemError &e){
		return e.code();
	}
}

bool getLine(StreamBuffer &buffer, std::string &line){
	line.clear();
	if(buffer.empty()){
		return false;
	}
	do {
		const int ch = buffer.get();
		if(ch == '\n'){
			break;
		} else if(ch == '\r'){
			if(buffer.peek() == '\n'){
				continue;
			}
			break;
		}
		line.push_back(ch);
	} while(!buffer.empty());
	return true;
}

}
