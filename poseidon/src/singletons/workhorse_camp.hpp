// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_SINGLETONS_WORKHORSE_CAMP_HPP_
#define POSEIDON_SINGLETONS_WORKHORSE_CAMP_HPP_

#include "../cxx_ver.hpp"
#include <boost/shared_ptr.hpp>
#include <boost/function.hpp>
#include <string>

namespace Poseidon {

class Promise;

class Workhorse_camp {
private:
	Workhorse_camp();

public:
	typedef std::function<void ()> Job_procedure;

	static void start();
	static void stop();

	static void enqueue_isolated(const boost::shared_ptr<Promise> &promise, Job_procedure procedure);
	// 具有相同 thread_hint 的任务保证由同一个线程执行。
	static void enqueue(const boost::shared_ptr<Promise> &promise, Job_procedure procedure, std::size_t thread_hint);
};

}

#endif
