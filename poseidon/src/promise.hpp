// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_PROMISE_HPP_
#define POSEIDON_PROMISE_HPP_

#include "cxx_ver.hpp"
#include "cxx_util.hpp"
#include "recursive_mutex.hpp"
#include <boost/shared_ptr.hpp>
#include <boost/type_traits/remove_const.hpp>
#include <boost/optional.hpp>

namespace Poseidon {

class Promise : NONCOPYABLE {
protected:
	mutable Recursive_mutex m_mutex;
	boost::optional<STD_EXCEPTION_PTR> m_except;

public:
	Promise()
		: m_mutex(), m_except()
	{
		//
	}
	virtual ~Promise();

public:
	bool is_satisfied() const NOEXCEPT;
	bool would_throw() const NOEXCEPT;
	void check_and_rethrow() const;

	void set_success(bool throw_if_already_set = true);
	void set_exception(STD_EXCEPTION_PTR except, bool throw_if_already_set = true);
};

template<typename ResultT>
class Promise_container : public Promise {
private:
	mutable boost::optional<typename boost::remove_const<ResultT>::type> m_result;
	bool m_result_accepted;

public:
	Promise_container()
		: m_result(), m_result_accepted(false)
	{
		//
	}
	~Promise_container() OVERRIDE;

public:
	ResultT * try_get() const NOEXCEPT {
		const Recursive_mutex::Unique_lock lock(m_mutex);
		if(Promise::would_throw()){
			return NULLPTR;
		}
		// `m_result_accepted` will not be false if `Promise::would_throw()` yields true.
		return m_result.get_ptr();
	}
	ResultT & get() const {
		const Recursive_mutex::Unique_lock lock(m_mutex);
		Promise::check_and_rethrow();
		// Likewise. See comments in `try_get()`.
		return m_result.get();
	}
	void set_success(typename boost::remove_const<ResultT>::type result, bool throw_if_already_set = true){
		const Recursive_mutex::Unique_lock lock(m_mutex);
		// If `m_result_accepted` is true, `Promise::set_success()` will throw an exception eventually. Hence we do not set the value here.
		if(!m_result_accepted){
			m_result = STD_MOVE_IDN(result);
		}
		Promise::set_success(throw_if_already_set);
		m_result_accepted = true;
	}
};

template<typename ResultT>
Promise_container<ResultT>::~Promise_container(){
	//
}

extern void yield(const boost::shared_ptr<const Promise> &promise, bool insignificant = true);

template<typename ResultT>
inline ResultT wait(const boost::shared_ptr<const Promise_container<ResultT> > &promise, bool insignificant = true){
	((yield))(promise, insignificant);
	return STD_MOVE(promise->get());
}
inline void wait(const boost::shared_ptr<const Promise> &promise, bool insignificant = true){
	((yield))(promise, insignificant);
}

}

#endif
