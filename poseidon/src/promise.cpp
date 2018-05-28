// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#include "precompiled.hpp"
#include "promise.hpp"
#include "exception.hpp"
#include "log.hpp"
#include "singletons/job_dispatcher.hpp"

namespace Poseidon {

Promise::~Promise(){
	//
}

bool Promise::is_satisfied() const NOEXCEPT {
	const std::lock_guard<std::recursive_mutex> lock(m_mutex);
	const STD_EXCEPTION_PTR *const ptr = m_except.get_ptr();
	return ptr;
}
bool Promise::would_throw() const NOEXCEPT {
	const std::lock_guard<std::recursive_mutex> lock(m_mutex);
	const STD_EXCEPTION_PTR *const ptr = m_except.get_ptr();
	return !ptr || *ptr;
}
void Promise::check_and_rethrow() const {
	const std::lock_guard<std::recursive_mutex> lock(m_mutex);
	const STD_EXCEPTION_PTR *const ptr = m_except.get_ptr();
	if(!ptr){
		POSEIDON_THROW(Exception, Rcnts::view("Promise has not been satisfied"));
	}
	if(*ptr){
		STD_RETHROW_EXCEPTION(*ptr);
	}
}

void Promise::set_success(bool throw_if_already_set){
	set_exception(STD_EXCEPTION_PTR(), throw_if_already_set);
}
void Promise::set_exception(STD_EXCEPTION_PTR except, bool throw_if_already_set){
	const std::lock_guard<std::recursive_mutex> lock(m_mutex);
	if(m_except){
		if(throw_if_already_set){
			POSEIDON_THROW(Exception, Rcnts::view("Promise has already been satisfied"));
		}
		return;
	}
	m_except = STD_MOVE_IDN(except);
}

void yield(const boost::shared_ptr<const Promise> &promise, bool insignificant){
	Job_dispatcher::yield(promise, insignificant);
}

}
