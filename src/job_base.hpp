// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_JOB_BASE_HPP_
#define POSEIDON_JOB_BASE_HPP_

#include "cxx_util.hpp"
#include <boost/weak_ptr.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/function.hpp>

namespace Poseidon {

class JobPromise;

class JobBase : NONCOPYABLE {
public:
	virtual ~JobBase();

public:
	// 如果一个任务被推迟执行且 Category 非空，
	// 则所有具有相同 Category 的后续任务都会被推迟，以维持其相对顺序。
	virtual boost::weak_ptr<const void> get_category() const = 0;
	virtual void perform() = 0;
};

// 加到全局队列中（外部不可见），线程安全的。
// 如果任务已加入队列之后执行 **withdraw = true 则会撤销该任务。
extern void enqueue_job(boost::shared_ptr<JobBase> job,
	boost::shared_ptr<const JobPromise> promise = boost::shared_ptr<const JobPromise>(),
	boost::shared_ptr<const bool> withdrawn = boost::shared_ptr<const bool>());
// 切到其它协程，直到 pred 为空或者 pred() 返回 true。必须在任务函数中调用。
extern void yield_job(boost::shared_ptr<const JobPromise> promise);
// 设定当前的协程不再可以 yield，此后再调用 yield_job 会抛出异常。
// 此操作不可逆。
extern void detach_yieldable_job() NOEXCEPT;

}

#endif
