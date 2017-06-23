// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2017, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_JOB_PROMISE_HPP_
#define POSEIDON_JOB_PROMISE_HPP_

#include "cxx_ver.hpp"
#include "cxx_util.hpp"
#include "recursive_mutex.hpp"
#include <boost/shared_ptr.hpp>

#ifdef POSEIDON_CXX11
#	include <exception>
#else
#	include <boost/exception_ptr.hpp>
#endif

namespace Poseidon {

class JobPromise : NONCOPYABLE {
protected:
	mutable RecursiveMutex m_mutex;
	bool m_satisfied;
#ifdef POSEIDON_CXX11
	std::exception_ptr m_except;
#else
	boost::exception_ptr m_except;
#endif

public:
	JobPromise() NOEXCEPT;
	virtual ~JobPromise();

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

template<typename T>
class JobPromiseContainer : public JobPromise {
private:
	T m_t;

public:
	JobPromiseContainer()
		: JobPromise()
		, m_t()
	{
	}
	explicit JobPromiseContainer(T t)
		: JobPromise()
		, m_t(STD_MOVE(t))
	{
	}

public:
	T *try_get() const NOEXCEPT {
		const RecursiveMutex::UniqueLock lock(m_mutex);
		if(JobPromise::would_throw()){
			return NULLPTR;
		}
		return const_cast<T *>(reinterpret_cast<const T *>(reinterpret_cast<const char (&)[1]>(m_t)));
	}
	T &get() const {
		const RecursiveMutex::UniqueLock lock(m_mutex);
		JobPromise::check_and_rethrow();
		return const_cast<T &>(m_t);
	}
	void set_success(T t){
		const RecursiveMutex::UniqueLock lock(m_mutex);
		m_t = STD_MOVE_IDN(t);
		JobPromise::set_success();
	}
};

template<typename T>
class JobPromiseContainer<const T> : public JobPromise {
private:
	T m_t;

public:
	JobPromiseContainer()
		: JobPromise()
		, m_t()
	{
	}
	explicit JobPromiseContainer(T t)
		: JobPromise()
		, m_t(STD_MOVE(t))
	{
	}

public:
	const T *try_get() const NOEXCEPT {
		const RecursiveMutex::UniqueLock lock(m_mutex);
		if(JobPromise::would_throw()){
			return NULLPTR;
		}
		return reinterpret_cast<const T *>(reinterpret_cast<const char (&)[1]>(m_t));
	}
	const T &get() const {
		const RecursiveMutex::UniqueLock lock(m_mutex);
		JobPromise::check_and_rethrow();
		return m_t;
	}
	void set_success(T t){
		const RecursiveMutex::UniqueLock lock(m_mutex);
		m_t = STD_MOVE_IDN(t);
		JobPromise::set_success();
	}
};

extern void yield(const boost::shared_ptr<const JobPromise> &promise, bool insignificant = true);

}

#endif
