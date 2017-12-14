// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2017, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_PROMISE_HPP_
#define POSEIDON_PROMISE_HPP_

#include "cxx_ver.hpp"
#include "cxx_util.hpp"
#include "recursive_mutex.hpp"
#include <boost/shared_ptr.hpp>
#include <boost/type_traits/remove_const.hpp>

#ifdef POSEIDON_CXX11
#	include <exception>
#	include <memory>
#else
#	include <boost/exception_ptr.hpp>
#endif

namespace Poseidon {

class Promise : NONCOPYABLE {
protected:
	mutable RecursiveMutex m_mutex;
	bool m_satisfied;
#ifdef POSEIDON_CXX11
	std::exception_ptr m_except;
#else
	boost::exception_ptr m_except;
#endif

public:
	Promise() NOEXCEPT;
	virtual ~Promise();

public:
	bool is_satisfied() const NOEXCEPT {
		const RecursiveMutex::UniqueLock lock(m_mutex);
		return m_satisfied;
	}
	bool would_throw() const NOEXCEPT;
	void check_and_rethrow() const;

	void set_success();
#ifdef POSEIDON_CXX11
	void set_exception(std::exception_ptr except);
#else
	void set_exception(boost::exception_ptr except);
#endif
};

template<typename ResultT>
class PromiseContainer : public Promise {
private:
	mutable typename boost::remove_const<ResultT>::type m_result;
	bool m_inited;

public:
	PromiseContainer()
		: Promise()
		, m_result(), m_inited(false)
	{ }
	explicit PromiseContainer(typename boost::remove_const<ResultT>::type result)
		: Promise()
		, m_result(STD_MOVE_IDN(result)), m_inited(false)
	{ }

public:
	ResultT *try_get() const NOEXCEPT {
		const RecursiveMutex::UniqueLock lock(m_mutex);
		if(Promise::would_throw()){
			return NULLPTR;
		}
		// Note that `m_inited` can't be `false` here once the promise is marked successful.
#ifdef POSEIDON_CXX11
		return std::addressof(m_result);
#else
		return reinterpret_cast<ResultT *>(reinterpret_cast<char (&)[1]>(m_result));
#endif
	}
	ResultT &get() const {
		const RecursiveMutex::UniqueLock lock(m_mutex);
		Promise::check_and_rethrow();
		return m_result;
	}
	void set_success(typename boost::remove_const<ResultT>::type result){
		const RecursiveMutex::UniqueLock lock(m_mutex);
		if(!m_inited){
			m_result = STD_MOVE_IDN(result);
		}
		Promise::set_success();
		m_inited = true;
	}
};

extern void yield(const boost::shared_ptr<const Promise> &promise, bool insignificant = true);

}

#endif
