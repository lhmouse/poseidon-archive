// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "promise.hpp"
#include "exception.hpp"
#include "../log.hpp"
#include "../atomic.hpp"

namespace Poseidon {

namespace {
	enum {
		S_LOCKED			= -1,
		S_UNSATISFIED		= 0,
		S_SATISFIED			= 1,
	};

	typedef Exception BasicException;
}

namespace MySql {
	Promise::Promise() NOEXCEPT
		: m_state(S_UNSATISFIED)
	{
	}
	Promise::~Promise(){
	}

	bool Promise::check(int cmp) const NOEXCEPT {
		for(;;){
			const int state = atomicLoad(m_state, ATOMIC_CONSUME);
			if(state != S_LOCKED){
				return state == cmp;
			}
			atomicPause();
		}
	}
	int Promise::lock() const NOEXCEPT {
		for(;;){
			const int state = atomicExchange(m_state, S_LOCKED, ATOMIC_SEQ_CST);
			if(state != S_LOCKED){
				return state;
			}
			atomicPause();
		}
	}
	void Promise::unlock(int state) const NOEXCEPT {
		atomicStore(m_state, state, ATOMIC_SEQ_CST);
	}

	bool Promise::isSatisfied() const NOEXCEPT {
		return !check(S_UNSATISFIED);
	}
	void Promise::checkAndRethrow() const {
		const int state = lock();
		if(state == S_UNSATISFIED){
			unlock(state);
			DEBUG_THROW(BasicException, sslit("MySQL promise is not satisfied"));
		}
		const AUTO(except, m_except);
		unlock(state);

		if(except){
			boost::rethrow_exception(m_except);
		}
	}

	void Promise::setSuccess() NOEXCEPT {
		const int state = lock();
		if(state != S_UNSATISFIED){
			unlock(state);
			DEBUG_THROW(BasicException, sslit("MySQL promise is already satisfied"));
		}
		m_except = boost::exception_ptr();
		unlock(S_SATISFIED);
	}
	void Promise::setException(const boost::exception_ptr &except) NOEXCEPT {
		assert(except);

		const int state = lock();
		if(state != S_UNSATISFIED){
			unlock(state);
			DEBUG_THROW(BasicException, sslit("MySQL promise is already satisfied"));
		}
		m_except = except;
		unlock(S_SATISFIED);
	}
}

}
