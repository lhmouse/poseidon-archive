// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#include "precompiled.hpp"
#include "raii.hpp"
#include <unistd.h>
#include <errno.h>
#include <string.h>
#include "log.hpp"
#include "utilities.hpp"
using namespace Poseidon;

void FileCloser::operator()(int fd) const NOEXCEPT {
	if(::close(fd) != 0){
		const AUTO(desc, getErrorDesc());
		LOG_POSEIDON_WARN("::close() has failed: ", desc.get());
	}
}
