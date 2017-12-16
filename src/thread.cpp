// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2017, LH_Mouse. All wrongs reserved.

#include "precompiled.hpp"
#include "thread.hpp"
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
{ }
Thread::Thread(boost::function<void ()> proc, const char *tag)
	: m_impl()
{
	m_impl = boost::make_shared<Impl>();
	std::strncpy(m_impl->tag, tag, sizeof(m_impl->tag));
	m_impl->proc.swap(proc);

	m_impl->self = m_impl; // 制造循环引用。
	try {
		int err_code = ::pthread_create(&(m_impl->handle), NULLPTR, &Impl::thread_proc, m_impl.get());
		DEBUG_THROW_UNLESS(err_code == 0, SystemException);
	} catch(...){
		m_impl->self.reset();
		throw;
	}
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
	DEBUG_THROW_UNLESS(m_impl, Exception, sslit("Attempting to join a non-joinable thread"));
	int err_code = ::pthread_join(m_impl->handle, NULLPTR);
	DEBUG_THROW_UNLESS(err_code == 0, SystemException);
	m_impl.reset();
}
void Thread::detach(){
	DEBUG_THROW_UNLESS(m_impl, Exception, sslit("Attempting to detach a non-joinable thread"));
	int err_code = ::pthread_detach(m_impl->handle);
	DEBUG_THROW_UNLESS(err_code == 0, SystemException);
	m_impl.reset();
}

}
