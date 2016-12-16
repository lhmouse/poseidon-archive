// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2016, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_ASYNC_JOB_HPP_
#define POSEIDON_ASYNC_JOB_HPP_

#include "cxx_ver.hpp"
#include "job_promise.hpp"
#include <boost/function.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/weak_ptr.hpp>

namespace Poseidon {

extern void enqueue_async_categorized_job(boost::weak_ptr<const void> category,
	boost::shared_ptr<JobPromise> promise, boost::function<void ()> proc,
	boost::shared_ptr<const bool> withdrawn = boost::shared_ptr<const bool>());
extern void enqueue_async_job(
	boost::shared_ptr<JobPromise> promise, boost::function<void ()> proc,
	boost::shared_ptr<const bool> withdrawn = boost::shared_ptr<const bool>());

}

#endif
