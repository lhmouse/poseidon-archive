// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2017, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_MODULE_RAII_HPP_
#define POSEIDON_MODULE_RAII_HPP_

#include "cxx_ver.hpp"
#include "cxx_util.hpp"
#include <boost/shared_ptr.hpp>
#include <boost/container/deque.hpp>

namespace Poseidon {

class HandleStack {
private:
	boost::container::deque<boost::shared_ptr<const void> > m_queue;

private:
	HandleStack(const HandleStack &);
	HandleStack &operator=(const HandleStack &);

public:
	HandleStack()
		: m_queue()
	{ }
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

/*
	MODULE_RAII(handles){
		AUTO(foo, boost::make_shared<Foo>());
		handles.push(STD_MOVE_IDN(foo));
	}
*/
#define MODULE_RAII_PRIORITY(handles_, priority_)	\
	namespace {	\
		namespace TOKEN_CAT3(ModuleRaii_, __LINE__, Stub_) {	\
			struct Stub_ : public ::Poseidon::ModuleRaiiBase {	\
				Stub_()	\
					: ::Poseidon::ModuleRaiiBase(priority_)	\
				{	\
				}	\
				void init(::Poseidon::HandleStack &) const FINAL;	\
			} const stub_;	\
		}	\
	}	\
	void TOKEN_CAT3(ModuleRaii_, __LINE__, Stub_)::Stub_::init(::Poseidon::HandleStack & handles_) const

#define MODULE_RAII(handles_)   MODULE_RAII_PRIORITY(handles_, 65535)

#endif
