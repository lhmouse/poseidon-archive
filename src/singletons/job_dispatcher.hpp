// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2017, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_SINGLETONS_JOB_DISPATCHER_HPP_
#define POSEIDON_SINGLETONS_JOB_DISPATCHER_HPP_

#include "../cxx_ver.hpp"
#include <boost/shared_ptr.hpp>

namespace Poseidon {

class JobBase;
class Promise;

class JobDispatcher {
private:
	JobDispatcher();

public:
	static void start();
	static void stop();

	static void do_modal(const volatile bool &running);

	static void enqueue(boost::shared_ptr<JobBase> job, boost::shared_ptr<const bool> withdrawn);
	static void yield(const boost::shared_ptr<const Promise> &promise, bool insignificant);
};

}

#endif
