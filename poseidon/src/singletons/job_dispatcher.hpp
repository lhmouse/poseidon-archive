// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_SINGLETONS_JOB_DISPATCHER_HPP_
#define POSEIDON_SINGLETONS_JOB_DISPATCHER_HPP_

#include "../cxx_ver.hpp"
#include <boost/shared_ptr.hpp>

namespace Poseidon {

class Job_base;
class Promise;

class Job_dispatcher {
private:
	Job_dispatcher();

public:
	static void start();
	static void stop();

	static void do_modal(const volatile bool &running);

	static void enqueue(boost::shared_ptr<Job_base> job, boost::shared_ptr<const bool> withdrawn);
	// Pass `promise` by value to avoid false aliasing.
	static void yield(boost::shared_ptr<const Promise> promise, bool insignificant);
};

}

#endif
