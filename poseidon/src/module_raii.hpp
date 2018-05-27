// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_MODULE_RAII_HPP_
#define POSEIDON_MODULE_RAII_HPP_

#include "cxx_ver.hpp"
#include "cxx_util.hpp"
#include "profiler.hpp"
#include <boost/shared_ptr.hpp>
#include <boost/container/deque.hpp>

namespace Poseidon {

class Handle_stack {
private:
	boost::container::deque<boost::shared_ptr<const void> > m_queue;

public:
	Handle_stack()
		: m_queue()
	{
		//
	}
#ifndef POSEIDON_CXX11
	Handle_stack(const Handle_stack &rhs)
		: m_queue(rhs.m_queue)
	{
		//
	}
	Handle_stack &operator=(const Handle_stack &rhs){
		m_queue = rhs.m_queue;
		return *this;
	}
#endif
	~Handle_stack();

public:
	void push(boost::shared_ptr<const void> handle);
	boost::shared_ptr<const void> pop();

	bool empty() const {
		return m_queue.empty();
	}
	std::size_t size() const {
		return m_queue.size();
	}
	void clear() NOEXCEPT; // 确保逆序析构。

	void swap(Handle_stack &rhs) NOEXCEPT {
		using std::swap;
		swap(m_queue, rhs.m_queue);
	}
};

inline void swap(Handle_stack &lhs, Handle_stack &rhs) NOEXCEPT {
	lhs.swap(rhs);
}

enum {
	module_init_priority_essential      =   100,
	module_init_priority_static         = 10000,
	module_init_priority_high           = 20000,
	module_init_priority_above_normal   = 30000,
	module_init_priority_normal         = 40000,
	module_init_priority_below_normal   = 50000,
	module_init_priority_low            = 60000,
};

class Module_raii_base : NONCOPYABLE {
public:
	explicit Module_raii_base(long priority);
	virtual ~Module_raii_base();

public:
	virtual void init(Handle_stack &handles) const = 0;
};

}

#define POSEIDON_MODULE_RAII_PRIORITY(handles_, priority_)	\
	namespace {	\
		struct POSEIDON_LAZY(POSEIDON_CAT2, Module_raii_stub_, __LINE__) FINAL : private ::Poseidon::Module_raii_base {	\
			static void unwrapped_init_(::Poseidon::Handle_stack &);	\
			/* constructor */	\
			explicit POSEIDON_LAZY(POSEIDON_CAT2, Module_raii_stub_, __LINE__)()	\
				: ::Poseidon::Module_raii_base(priority_)	\
			{ }	\
			/* overriden virtual function */	\
			void init(::Poseidon::Handle_stack &handle_stack_) const {	\
				POSEIDON_PROFILE_ME;	\
				unwrapped_init_(handle_stack_);	\
			}	\
		} const POSEIDON_LAZY(POSEIDON_CAT2, module_raii_stub_, __LINE__);	\
	}	\
	void POSEIDON_LAZY(POSEIDON_CAT2, Module_raii_stub_, __LINE__)::unwrapped_init_(::Poseidon::Handle_stack &handles_)

#define POSEIDON_MODULE_RAII(handles_)   POSEIDON_MODULE_RAII_PRIORITY(handles_, ::Poseidon::module_init_priority_normal)

#endif
