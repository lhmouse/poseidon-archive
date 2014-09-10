#include "../precompiled.hpp"
#include "raii.hpp"
#include <unistd.h>
#include <errno.h>
#include <string.h>
#include "log.hpp"
#include "utilities.hpp"
using namespace Poseidon;

namespace Poseidon {

void closeFile(int fd) NOEXCEPT {
	if(::close(fd) != 0){
		const AUTO(desc, getErrorDesc());
		LOG_WARNING("::close() has failed: ", desc.get());
	}
}

}
