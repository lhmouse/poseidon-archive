// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#include "precompiled.hpp"
#include "thread.hpp"
#include "log.hpp"
#include "system_exception.hpp"

namespace Poseidon {

namespace {
	struct ThreadControlBlock {
		boost::function<void ()> proc;
		SharedNts tag;
		SharedNts name;

		boost::shared_ptr<ThreadControlBlock> ref;
		::pthread_t handle;
	};

	void *thread_proc(void *param)
	try {
		boost::shared_ptr<ThreadControlBlock> tcb;
		// Break the circular reference. This allows `*tcb` to be deleted.
		tcb.swap(*static_cast<boost::shared_ptr<ThreadControlBlock> *>(param));

		// Ignore any errors.
		Logger::set_thread_tag(tcb->tag);
		::pthread_setname_np(::pthread_self(), tcb->name);

		// Do something.
		tcb->proc();
		return NULLPTR;
	} catch(...){
		std::terminate();
	}
}

Thread::Thread(boost::function<void ()> proc, SharedNts tag, SharedNts name){
	ThreadControlBlock temp = { STD_MOVE_IDN(proc), STD_MOVE(tag), STD_MOVE(name) };
	const AUTO(tcb, boost::make_shared<ThreadControlBlock>(STD_MOVE(temp)));
	try {
		// Create a circular reference. This prevents `*tcb` from being deleted before it is consumed by the new thread.
		int err = ::pthread_create(&(tcb->handle), NULLPTR, &thread_proc, &(tcb->ref = tcb));
		DEBUG_THROW_UNLESS(err == 0, SystemException, err);
	} catch(...){
		// Break the circular reference. This allows `*tcb` to be deleted.
		tcb->ref.reset();
		throw;
	}
	m_tcb = tcb;
}
Thread::~Thread(){
	if(joinable()){
		LOG_POSEIDON_FATAL("Attempting to destroy a joinable thread.");
		std::terminate();
	}
}

bool Thread::joinable() const NOEXCEPT {
	const AUTO(tcb, static_cast<ThreadControlBlock *>(m_tcb.get()));
	return tcb;
}
void Thread::join(){
	const AUTO(tcb, static_cast<ThreadControlBlock *>(m_tcb.get()));
	DEBUG_THROW_UNLESS(tcb, Exception, sslit("Thread is not joinable"));
	int err = ::pthread_join(tcb->handle, NULLPTR);
	DEBUG_THROW_UNLESS(err == 0, SystemException, err);
	m_tcb.reset();
}

}
