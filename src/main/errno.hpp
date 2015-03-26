// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_ERRNO_HPP_
#define POSEIDON_ERRNO_HPP_

#include "cxx_ver.hpp"
#include <string>
#include <errno.h>
#include "shared_nts.hpp"

namespace Poseidon {

SharedNts getErrorDesc(int errCode = errno) NOEXCEPT;
std::string getErrorDescAsString(int errCode = errno);

}

#endif
