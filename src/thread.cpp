// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2016, LH_Mouse. All wrongs reserved.

#include "precompiled.hpp"
#include "thread.hpp"
#include <pthread.h>
#include "log.hpp"
#include "system_exception.hpp"

namespace Poseidon {

struct Thread::Impl {
	static void *thread_proc(void *impl){
		boost::shared_ptr<Impl> self;
		self.swap(static_cast<Impl *>(impl)->self);

		Logger::set_thread_tag(self->tag);

		try {
			self->proc();
		} catch(...){
			std::terminate();
		}
		return NULLPTR;
	}

	char tag[8];
	boost::function<void ()> proc;
	::pthread_t handle;

	boost::shared_ptr<Impl> self;
};

Thread::Thread() NOEXCEPT
	: m_impl()
{
}
Thread::Thread(boost::function<void ()> proc, const char *tag)
	: m_impl()
{
	m_impl = boost::make_shared<Impl>();
	std::strncpy(m_impl->tag, tag, sizeof(m_impl->tag));
	m_impl->proc.swap(proc);

	m_impl->self = m_impl; // 制造循环引用。
	try {
		const int err = ::pthread_create(&(m_impl->handle), NULLPTR, &Impl::thread_proc, m_impl.get());
		if(err != 0){
			LOG_POSEIDON_ERROR("::pthread_create() failed with error code ", err);
			DEBUG_THROW(SystemException, err);
		}
	} catch(...){
		m_impl->self.reset();
		throw;
	}
}
Thread::Thread(Move<Thread> rhs) NOEXCEPT {
	rhs.swap(*this);
}
Thread &Thread::operator=(Move<Thread> rhs) NOEXCEPT {
	rhs.swap(*this);
	return *this;
}
Thread::~Thread(){
	if(joinable()){
		LOG_POSEIDON_FATAL("Destructing a joinable thread.");
		std::terminate();
	}
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

	::pthread_detach(m_impl->handle);
	m_impl.reset();
}

}
