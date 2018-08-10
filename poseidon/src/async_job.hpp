// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_ASYNC_JOB_HPP_
#define POSEIDON_ASYNC_JOB_HPP_

#include "cxx_ver.hpp"
#include "job_base.hpp"
#include "promise.hpp"
#include <boost/shared_ptr.hpp>
#include <boost/weak_ptr.hpp>

namespace Poseidon {

template<typename FunctionT>
class Async_job_wrapper : public Job_base {
private:
	const boost::weak_ptr<const void> m_category;
	const boost::weak_ptr<Promise> m_weak_promise;
	FunctionT m_func;

public:
	Async_job_wrapper(boost::weak_ptr<const void> category, const boost::shared_ptr<Promise> &promise, FunctionT func)
		: m_category(STD_MOVE(category))
		, m_weak_promise(promise), m_func(STD_MOVE_IDN(func))
	{
	}
	~Async_job_wrapper() OVERRIDE;

protected:
	boost::weak_ptr<const void> get_category() const FINAL {
		return m_category;
	}
	void perform() FINAL {
		STD_EXCEPTION_PTR except;
		try {
			m_func();
		} catch(std::exception &e){
			except = STD_CURRENT_EXCEPTION();
		} catch(...){
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

template<typename FunctionT>
Async_job_wrapper<FunctionT>::~Async_job_wrapper(){ }


template<typename FunctionT>
inline void enqueue_async_categorized_job(boost::weak_ptr<const void> category, const boost::shared_ptr<Promise> &promise, FunctionT func,
	boost::shared_ptr<const bool> withdrawn = boost::shared_ptr<const bool>())
{
	const AUTO(job, boost::make_shared<Async_job_wrapper<FunctionT> >(STD_MOVE(category), promise, STD_MOVE_IDN(func)));
	enqueue(job, STD_MOVE_IDN(withdrawn));
}
template<typename FunctionT>
inline void enqueue_async_job(const boost::shared_ptr<Promise> &promise, FunctionT func,
	boost::shared_ptr<const bool> withdrawn = boost::shared_ptr<const bool>())
{
	const AUTO(job, boost::make_shared<Async_job_wrapper<FunctionT> >(boost::weak_ptr<const void>(), promise, STD_MOVE_IDN(func)));
	enqueue(job, STD_MOVE_IDN(withdrawn));
}

}

#endif
