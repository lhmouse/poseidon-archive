// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_ASYNC_JOB_BASE_HPP_
#define POSEIDON_ASYNC_JOB_BASE_HPP_

#include "cxx_util.hpp"
#include <boost/function.hpp>

namespace Poseidon {

extern void enqueueAsyncJob(boost::function<void ()> proc);

}

#endif
