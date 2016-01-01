// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2016, LH_Mouse. All wrongs reserved.

#include "precompiled.hpp"
#include "file.hpp"
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include "raii.hpp"
#include "system_exception.hpp"

namespace Poseidon {

void file_get_contents(StreamBuffer &contents, const char *path){
	const UniqueFile file(::open(path, O_RDONLY));
	if(!file){
		DEBUG_THROW(SystemException);
	}
	StreamBuffer temp;
	for(;;){
		char read_buf[4096];
		const ::ssize_t bytes_read = ::read(file.get(), read_buf, sizeof(read_buf));
		if(bytes_read < 0){
			DEBUG_THROW(SystemException);
		} else if(bytes_read == 0){
			break;
		}
		temp.put(read_buf, static_cast<std::size_t>(bytes_read));
	}
	contents.splice(temp);
}
int file_get_contents_nothrow(StreamBuffer &contents, const char *path){
	try {
		file_get_contents(contents, path);
		return 0;
	} catch(SystemException &e){
		return e.code();
	}
}

void file_put_contents(const char *path, StreamBuffer contents, bool append){
	const UniqueFile file(::open(path, O_WRONLY | O_CREAT | (append ? O_APPEND : O_TRUNC)));
	if(!file){
		DEBUG_THROW(SystemException);
	}
	for(;;){
		char write_buf[4096];
		const std::size_t bytes_to_write = contents.get(write_buf, sizeof(write_buf));
		if(bytes_to_write == 0){
			break;
		}
		if(::write(file.get(), write_buf, bytes_to_write) < static_cast< ::ssize_t>(bytes_to_write)){
			DEBUG_THROW(SystemException);
		}
	}
}
int file_put_contents_nothrow(const char *path, StreamBuffer contents, bool append){
	try {
		file_put_contents(path, STD_MOVE(contents), append);
		return 0;
	} catch(SystemException &e){
		return e.code();
	}
}

}
