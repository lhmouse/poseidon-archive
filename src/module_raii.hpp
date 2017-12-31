// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2017, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_MODULE_RAII_HPP_
#define POSEIDON_MODULE_RAII_HPP_

#include "cxx_ver.hpp"
#include "cxx_util.hpp"
#include "profiler.hpp"
#include <boost/shared_ptr.hpp>
#include <boost/container/deque.hpp>

namespace Poseidon {

class HandleStack {
private:
	boost::container::deque<boost::shared_ptr<const void> > m_queue;

public:
	HandleStack()
		: m_queue()
	{ }
#ifndef POSEIDON_CXX11
	HandleStack(const HandleStack &rhs)
		: m_queue(rhs.m_queue)
	{ }
	HandleStack &operator=(const HandleStack &rhs){
		m_queue = rhs.m_queue;
		return *this;
	}
#endif
	~HandleStack();

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

	void swap(HandleStack &rhs) NOEXCEPT {
		using std::swap;
		swap(m_queue, rhs.m_queue);
	}
};

inline void swap(HandleStack &lhs, HandleStack &rhs) NOEXCEPT {
	lhs.swap(rhs);
}

class ModuleRaiiBase : NONCOPYABLE {
public:
	explicit ModuleRaiiBase(long priority);
	virtual ~ModuleRaiiBase();

public:
	virtual void init(HandleStack &handles) const = 0;
};

}

#define INIT_PRIORITY_ESSENTIAL        100
#define INIT_PRIORITY_STATIC         10000
#define INIT_PRIORITY_HIGH           20000
#define INIT_PRIORITY_ABOVE_NORMAL   30000
#define INIT_PRIORITY_NORMAL         40000
#define INIT_PRIORITY_BELOW_NORMAL   50000
#define INIT_PRIORITY_LOW            60000

#define MODULE_RAII_PRIORITY(handles_, priority_)	\
	namespace {	\
		namespace TOKEN_CAT3(ModuleRaii_, __LINE__, Stub_) {	\
			struct Stub_ : public ::Poseidon::ModuleRaiiBase {	\
				Stub_()	\
					: ::Poseidon::ModuleRaiiBase(priority_)	\
				{ }	\
				void init(::Poseidon::HandleStack & handle_stack_) const FINAL {	\
					PROFILE_ME;	\
					unwrapped_init_(handle_stack_);	\
				}	\
				void unwrapped_init_(::Poseidon::HandleStack &) const;	\
			} const stub_;	\
		}	\
	}	\
	void TOKEN_CAT3(ModuleRaii_, __LINE__, Stub_)::Stub_::unwrapped_init_(::Poseidon::HandleStack & handles_) const

#define MODULE_RAII(handles_)   MODULE_RAII_PRIORITY(handles_, INIT_PRIORITY_NORMAL)


#endif
