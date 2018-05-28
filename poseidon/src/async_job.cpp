// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#include "precompiled.hpp"
#include "async_job.hpp"
#include "job_base.hpp"
#include "exception.hpp"
#include "log.hpp"
#include "profiler.hpp"
#include "singletons/job_dispatcher.hpp"

namespace Poseidon {

namespace {
	class Async_job : public Job_base {
	private:
		const boost::weak_ptr<const void> m_category;
		const boost::weak_ptr<Promise> m_weak_promise;
		const std::function<void ()> m_procedure;

	public:
		Async_job(boost::weak_ptr<const void> category, const boost::shared_ptr<Promise> &promise, std::function<void ()> procedure)
			: m_category(STD_MOVE(category))
			, m_weak_promise(promise), m_procedure(STD_MOVE_IDN(procedure))
		{
			//
		}

	protected:
		boost::weak_ptr<const void> get_category() const FINAL {
			return m_category;
		}
		void perform() FINAL {
			POSEIDON_PROFILE_ME;

			STD_EXCEPTION_PTR except;
			try {
				m_procedure();
			} catch(std::exception &e){
				POSEIDON_LOG_DEBUG("std::exception thrown: what = ", e.what());
				except = STD_CURRENT_EXCEPTION();
			} catch(...){
				POSEIDON_LOG_DEBUG("Unknown exception thrown.");
				except = STD_CURRENT_EXCEPTION();
			}
			const AUTO(promise, m_weak_promise.lock());
			if(promise){
				if(except){
					promise->set_exception(STD_MOVE(except), false);
				} else {
					promise->set_success(false);
				}
			}
		}
	};
}

void enqueue_async_categorized_job(boost::weak_ptr<const void> category, const boost::shared_ptr<Promise> &promise, std::function<void ()> procedure, boost::shared_ptr<const bool> withdrawn){
	Job_dispatcher::enqueue(boost::make_shared<Async_job>(STD_MOVE(category), promise, STD_MOVE_IDN(procedure)), STD_MOVE(withdrawn));
}
void enqueue_async_job(const boost::shared_ptr<Promise> &promise, std::function<void ()> procedure, boost::shared_ptr<const bool> withdrawn){
	Job_dispatcher::enqueue(boost::make_shared<Async_job>(boost::weak_ptr<const void>(), promise, STD_MOVE_IDN(procedure)), STD_MOVE(withdrawn));
}

}
