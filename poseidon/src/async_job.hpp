// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_ASYNC_JOB_HPP_
#define POSEIDON_ASYNC_JOB_HPP_

#include "cxx_ver.hpp"
#include "promise.hpp"
#include <boost/function.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/weak_ptr.hpp>

namespace Poseidon {

extern void enqueue_async_categorized_job(boost::weak_ptr<const void> category, const boost::shared_ptr<Promise> &promise, std::function<void ()> procedure, boost::shared_ptr<const bool> withdrawn = boost::shared_ptr<const bool>());
extern void enqueue_async_job(const boost::shared_ptr<Promise> &promise, std::function<void ()> procedure, boost::shared_ptr<const bool> withdrawn = boost::shared_ptr<const bool>());

}

#endif
