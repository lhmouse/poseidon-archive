// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#include "precompiled.hpp"
#include "thread.hpp"
#include <pthread.h>
#include "log.hpp"
#include "system_exception.hpp"

namespace Poseidon {

struct Thread::Impl {
	static void *threadProc(void *impl){
		boost::shared_ptr<Impl> self;
		self.swap(static_cast<Impl *>(impl)->self);

		try {
			self->proc();
		} catch(...){
			std::terminate();
		}
		return NULLPTR;
	}

	boost::function<void ()> proc;
	pthread_t handle;

	boost::shared_ptr<Impl> self;
};

Thread::Thread() NOEXCEPT {
}
Thread::Thread(boost::function<void ()> proc){
	m_impl = boost::make_shared<Impl>();
	m_impl->proc.swap(proc);
	const int err = ::pthread_create(&(m_impl->handle), NULLPTR, &Impl::threadProc, m_impl.get());
	if(err != 0){
		LOG_POSEIDON_ERROR("::pthread_create() failed with error code ", err);
		DEBUG_THROW(SystemException, err);
	}
	m_impl->self = m_impl;
}
Thread::~Thread(){
	if(joinable()){
		LOG_POSEIDON_FATAL("Destructing a joinable thread.");
		std::terminate();
	}
}

void Thread::swap(Thread &rhs) NOEXCEPT {
	std::swap(m_impl, rhs.m_impl);
}

bool Thread::joinable() const NOEXCEPT {
	return !!m_impl;
}
void Thread::join(){
	assert(m_impl);

	::pthread_join(m_impl->handle, NULLPTR);
	m_impl.reset();
}
void Thread::detach(){
	assert(m_impl);

	m_impl.reset();
}

}
