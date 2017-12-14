// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2017, LH_Mouse. All wrongs reserved.

#include "precompiled.hpp"
#include "async_job.hpp"
#include "job_base.hpp"
#include "exception.hpp"
#include "log.hpp"
#include "profiler.hpp"
#include "singletons/job_dispatcher.hpp"

namespace Poseidon {

namespace {
	class AsyncJob : public JobBase {
	private:
		const boost::weak_ptr<const void> m_category;
		const boost::weak_ptr<JobPromise> m_weak_promise;
		const boost::function<void ()> m_procedure;

	public:
		AsyncJob(boost::weak_ptr<const void> category, const boost::shared_ptr<JobPromise> &promise, boost::function<void ()> procedure)
			: m_category(STD_MOVE(category))
			, m_weak_promise(promise), m_procedure(STD_MOVE_IDN(procedure))
		{ }

	protected:
		boost::weak_ptr<const void> get_category() const OVERRIDE {
			return m_category;
		}
		void perform() OVERRIDE {
			PROFILE_ME;

#ifdef POSEIDON_CXX11
			std::exception_ptr except;
#else
			boost::exception_ptr except;
#endif
			try {
				m_procedure();
			} catch(Exception &e){
				LOG_POSEIDON_DEBUG("Exception thrown: what = ", e.what());
#ifdef POSEIDON_CXX11
				except = std::current_exception();
#else
				except = boost::copy_exception(e);
#endif
			} catch(std::exception &e){
				LOG_POSEIDON_DEBUG("std::exception thrown: what = ", e.what());
#ifdef POSEIDON_CXX11
				except = std::current_exception();
#else
				except = boost::copy_exception(std::runtime_error(e.what()));
#endif
			} catch(...){
				LOG_POSEIDON_DEBUG("Unknown exception thrown.");
#ifdef POSEIDON_CXX11
				except = std::current_exception();
#else
				except = boost::copy_exception(std::bad_exception());
#endif
			}
			const AUTO(promise, m_weak_promise.lock());
			if(promise && !promise->is_satisfied()){
				try {
					if(!except){
						promise->set_success();
					} else {
						promise->set_exception(except);
					}
				} catch(std::exception &e){
					LOG_POSEIDON_ERROR("std::exception thrown: what = ", e.what());
				}
			}
		}
	};
}

void enqueue_async_categorized_job(boost::weak_ptr<const void> category, const boost::shared_ptr<JobPromise> &promise, boost::function<void ()> procedure,
	boost::shared_ptr<const bool> withdrawn)
{
	AUTO(job, boost::make_shared<AsyncJob>(STD_MOVE(category), promise, STD_MOVE_IDN(procedure)));
	JobDispatcher::enqueue(STD_MOVE_IDN(job), STD_MOVE(withdrawn));
}
void enqueue_async_job(const boost::shared_ptr<JobPromise> &promise, boost::function<void ()> procedure,
	boost::shared_ptr<const bool> withdrawn)
{
	enqueue_async_categorized_job(VAL_INIT, promise, STD_MOVE_IDN(procedure), STD_MOVE(withdrawn));
}

}
