// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_ERRNO_HPP_
#define POSEIDON_ERRNO_HPP_

#include "cxx_ver.hpp"
#include "shared_nts.hpp"

namespace Poseidon {

extern Shared_nts get_error_desc(int err_code = errno) NOEXCEPT;
extern std::string get_error_desc_as_string(int err_code = errno);

}

#endif
