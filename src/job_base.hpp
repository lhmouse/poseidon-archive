// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_JOB_BASE_HPP_
#define POSEIDON_JOB_BASE_HPP_

#include "cxx_util.hpp"
#include <exception>
#include <boost/weak_ptr.hpp>
#include <boost/shared_ptr.hpp>

namespace Poseidon {

class JobBase : NONCOPYABLE {
public:
	// 推迟当前任务执行。
	class TryAgainLater : public std::exception {
	private:
		boost::shared_ptr<const void> m_context;

	public:
		explicit TryAgainLater(boost::shared_ptr<const void> context) NOEXCEPT;
		~TryAgainLater() NOEXCEPT;

	public:
		const char *what() const NOEXCEPT OVERRIDE {
			return "Poseidon::JobBase::TryAgainLater";
		}

		const boost::shared_ptr<const void> &getContext() const NOEXCEPT {
			return m_context;
		}
	};

public:
	virtual ~JobBase();

public:
	// 如果一个任务被推迟执行且 Category 非空，
	// 则所有具有相同 Category 的后续任务都会被推迟，以维持其相对顺序。
	virtual boost::weak_ptr<const void> getCategory() const = 0;
	virtual void perform() const = 0;
};

// 加到全局队列中（外部不可见），线程安全的。
// 如果任务已加入队列之后执行 **withdraw = true 则会撤销该任务。
extern void enqueueJob(boost::shared_ptr<const JobBase> job, boost::uint64_t delay = 0,
	boost::shared_ptr<const bool> withdrawn = boost::shared_ptr<const bool>());
// 推迟当前任务执行，实际上会抛出一个异常。必须在任务函数中调用。
extern void suspendCurrentJob(boost::shared_ptr<const void> context) __attribute__((__noreturn__));

}

#endif
