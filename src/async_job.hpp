// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_ASYNC_JOB_BASE_HPP_
#define POSEIDON_ASYNC_JOB_BASE_HPP_

#include "cxx_ver.hpp"
#include <boost/function.hpp>
#include <boost/cstdint.hpp>

namespace Poseidon {

class JobPromise;

extern void enqueueAsyncJob(boost::function<void ()> proc,
	boost::shared_ptr<const JobPromise> promise = boost::shared_ptr<const JobPromise>(),
	boost::shared_ptr<const bool> withdrawn = boost::shared_ptr<const bool>());
extern void enqueueAsyncJob(boost::weak_ptr<const void> category, boost::function<void ()> proc,
	boost::shared_ptr<const JobPromise> promise = boost::shared_ptr<const JobPromise>(),
	boost::shared_ptr<const bool> withdrawn = boost::shared_ptr<const bool>());

}

#endif
