// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_ASYNC_JOB_BASE_HPP_
#define POSEIDON_ASYNC_JOB_BASE_HPP_

#include <boost/function.hpp>
#include <boost/cstdint.hpp>

namespace Poseidon {

extern void enqueueAsyncJob(boost::function<void ()> proc, boost::uint64_t delay = 0,
	boost::shared_ptr<bool> *withdrawn = NULLPTR);

}

#endif
