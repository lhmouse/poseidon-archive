#include "../precompiled.hpp"
#include "raii.hpp"
#include <unistd.h>
#include <errno.h>
#include <string.h>
#include "log.hpp"
using namespace Poseidon;

namespace Poseidon {

void closeFile(int fd) throw() {
	if(::close(fd) != 0){
		const int code = errno;
		char temp[256];
		const char *const reason = ::strerror_r(code, temp, sizeof(temp));
		LOG_WARNING,"::close() has failed, errno = ",code,": ",reason;
	}
}

}
