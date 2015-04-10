// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#include "precompiled.hpp"
#include "async_job.hpp"
#include "job_base.hpp"

namespace Poseidon {

namespace {
	class AsyncJob : public JobBase {
	private:
		const boost::function<void ()> m_proc;

	public:
		explicit AsyncJob(boost::function<void ()> proc)
			: m_proc(STD_MOVE_IDN(proc))
		{
		}

	public:
		boost::weak_ptr<const void> getCategory() const OVERRIDE {
			return VAL_INIT;
		}
		void perform() const OVERRIDE {
			m_proc();
		}
	};
}

void enqueueAsyncJob(boost::function<void ()> proc, boost::uint64_t delay,
	boost::shared_ptr<const bool> withdrawn)
{
	enqueueJob(boost::make_shared<AsyncJob>(STD_MOVE(proc)), delay, STD_MOVE(withdrawn));
}

}
