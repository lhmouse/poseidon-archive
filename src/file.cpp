// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#include "precompiled.hpp"
#include "file.hpp"
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include "raii.hpp"
#include "system_exception.hpp"

namespace Poseidon {

void fileGetContents(StreamBuffer &contents, const char *path){
	const UniqueFile file(::open(path, O_RDONLY));
	if(!file){
		DEBUG_THROW(SystemException);
	}
	StreamBuffer temp;
	for(;;){
		char readBuf[4096];
		const ::ssize_t bytesRead = ::read(file.get(), readBuf, sizeof(readBuf));
		if(bytesRead < 0){
			DEBUG_THROW(SystemException);
		} else if(bytesRead == 0){
			break;
		}
		temp.put(readBuf, static_cast<std::size_t>(bytesRead));
	}
	contents.splice(temp);
}
int fileGetContentsNoThrow(StreamBuffer &contents, const char *path){
	try {
		fileGetContents(contents, path);
		return 0;
	} catch(SystemException &e){
		return e.code();
	}
}

void filePutContents(const char *path, StreamBuffer contents, bool append){
	const UniqueFile file(::open(path, O_WRONLY | O_CREAT | (append ? O_APPEND : O_TRUNC)));
	if(!file){
		DEBUG_THROW(SystemException);
	}
	for(;;){
		char writeBuf[4096];
		const std::size_t bytesToWrite = contents.get(writeBuf, sizeof(writeBuf));
		if(bytesToWrite == 0){
			break;
		}
		if(::write(file.get(), writeBuf, bytesToWrite) < static_cast< ::ssize_t>(bytesToWrite)){
			DEBUG_THROW(SystemException);
		}
	}
}
int filePutContentsNoThrow(const char *path, StreamBuffer contents, bool append){
	try {
		filePutContents(path, STD_MOVE(contents), append);
		return 0;
	} catch(SystemException &e){
		return e.code();
	}
}

}
