#ifndef POSEIDON_RAII_HPP_
#define POSEIDON_RAII_HPP_

#include "../cxx_ver.hpp"

namespace Poseidon {

template<typename HandleT, typename CloserT>
class ScopedHandle {
private:
	HandleT m_handle;

public:
	ScopedHandle() NOEXCEPT
		: m_handle(CloserT()())
	{
	}
	explicit ScopedHandle(HandleT handle) NOEXCEPT
		: m_handle(CloserT()())
	{
		reset(handle);
	}
	ScopedHandle &operator=(HandleT handle) NOEXCEPT {
		reset(handle);
		return *this;
	}
	~ScopedHandle() NOEXCEPT {
		reset();
	}

private:
	ScopedHandle(const ScopedHandle &);
	void operator=(const ScopedHandle &);

public:
	HandleT get() const NOEXCEPT {
		return m_handle;
	}
	HandleT release() NOEXCEPT {
		const HandleT ret = m_handle;
		m_handle = CloserT()();
		return ret;
	}
	void reset(HandleT handle = CloserT()()) NOEXCEPT {
		const HandleT old = m_handle;
		m_handle = handle;
		if(old != CloserT()()){
			CloserT()(old);
		}
	}
	void reset(ScopedHandle &rhs) NOEXCEPT {
		if(&rhs == this){
			return;
		}
		swap(rhs);
		rhs.reset();
	}
	void swap(ScopedHandle &rhs) NOEXCEPT {
		const HandleT my = m_handle;
		m_handle = rhs.m_handle;
		rhs.m_handle = my;
	}
#ifdef POSEIDON_CXX11
	explicit
#endif
	operator bool() const NOEXCEPT {
		return get() != CloserT()();
	}
};

template<typename HandleT, typename CloserT>
class SharedHandle {
private:
	typedef ScopedHandle<HandleT, CloserT> Scoped;

	struct Control {
		Scoped h;
		volatile unsigned long ref;
	};

private:
	Control *m_control;

public:
	SharedHandle() NOEXCEPT
		: m_control(0)
	{
	}
	explicit SharedHandle(HandleT handle)
		: m_control(0)
	{
		reset(handle);
	}
	SharedHandle(const SharedHandle &rhs) NOEXCEPT
		: m_control(0)
	{
		reset(rhs);
	}
	SharedHandle &operator=(HandleT handle){
		Scoped tmp(handle);
		reset(tmp);
		return *this;
	}
	SharedHandle &operator=(const SharedHandle &rhs) NOEXCEPT {
		reset(rhs);
		return *this;
	}
	~SharedHandle() NOEXCEPT {
		reset();
	}

public:
	HandleT get() const NOEXCEPT {
		if(!m_control){
			return CloserT()();
		}
		return m_control->h.get();
	}
	bool unique() const NOEXCEPT {
		if(!m_control){
			return false;
		}
		volatile int barrier;
		__sync_lock_test_and_set(&barrier, 1);
		return m_control->ref == 1;
	}
	void reset() NOEXCEPT {
		if(!m_control){
			return;
		}
		if(__sync_sub_and_fetch(&m_control->ref, 1) == 0){
			delete m_control;
		}
		m_control = 0;
	}
	void reset(HandleT handle){
		Scoped tmp(handle);
		reset(tmp);
	}
	void reset(Scoped &scoped){
		if(!scoped){
			reset();
			return;
		}
		if(unique()){
			m_control->h.swap(scoped);
			scoped.reset();
		} else {
			reset();
			m_control = new Control;
			m_control->h.swap(scoped);
			m_control->ref = 1;
			volatile int barrier;
			__sync_lock_release(&barrier);
		}
	}
	void reset(const SharedHandle &rhs) NOEXCEPT {
		if(&rhs == this){
			return;
		}
		if(rhs.m_control){
			m_control = rhs.m_control;
			__sync_add_and_fetch(&m_control->ref, 1);
		} else {
			reset();
		}
	}
	void swap(SharedHandle &rhs) NOEXCEPT {
		Control *const my = m_control;
		m_control = rhs.m_control;
		rhs.m_control = my;
	}
#ifdef POSEIDON_CXX11
	explicit
#endif
	operator bool() const NOEXCEPT {
		return get() != CloserT()();
	}
};

template<typename HandleT, typename CloserT>
void swap(ScopedHandle<HandleT, CloserT> &lhs,
	ScopedHandle<HandleT, CloserT> &rhs) NOEXCEPT
{
	lhs.swap(rhs);
}

template<typename HandleT, typename CloserT>
void swap(SharedHandle<HandleT, CloserT> &lhs,
	SharedHandle<HandleT, CloserT> &rhs) NOEXCEPT
{
	lhs.swap(rhs);
}

#define DEFINE_RATIONAL_OPERATOR_(temp_, op_)	\
	template<typename HandleT, typename CloserT>	\
	bool operator op_(const temp_<HandleT, CloserT> &lhs,	\
		const temp_<HandleT, CloserT> &rhs) NOEXCEPT	\
	{	\
		return lhs.get() op_ rhs.get();	\
	}	\
	template<typename HandleT, typename CloserT>	\
	bool operator op_(HandleT lhs,	\
		const temp_<HandleT, CloserT> &rhs) NOEXCEPT	\
	{	\
		return lhs op_ rhs.get();	\
	}	\
	template<typename HandleT, typename CloserT>	\
	bool operator op_(const temp_<HandleT, CloserT> &lhs,	\
		HandleT rhs) NOEXCEPT	\
	{	\
		return lhs.get() op_ rhs;	\
	}

DEFINE_RATIONAL_OPERATOR_(ScopedHandle, ==)
DEFINE_RATIONAL_OPERATOR_(ScopedHandle, !=)
DEFINE_RATIONAL_OPERATOR_(ScopedHandle, <)
DEFINE_RATIONAL_OPERATOR_(ScopedHandle, >)
DEFINE_RATIONAL_OPERATOR_(ScopedHandle, <=)
DEFINE_RATIONAL_OPERATOR_(ScopedHandle, >=)

DEFINE_RATIONAL_OPERATOR_(SharedHandle, ==)
DEFINE_RATIONAL_OPERATOR_(SharedHandle, !=)
DEFINE_RATIONAL_OPERATOR_(SharedHandle, <)
DEFINE_RATIONAL_OPERATOR_(SharedHandle, >)
DEFINE_RATIONAL_OPERATOR_(SharedHandle, <=)
DEFINE_RATIONAL_OPERATOR_(SharedHandle, >=)

#undef DEFINE_RATIONAL_OPERATOR_

extern void closeFile(int fd) NOEXCEPT;

struct FileCloserT {
	int operator()() const NOEXCEPT {
		return -1;
	}
	void operator()(int fd) const NOEXCEPT {
		closeFile(fd);
	}
};

typedef ScopedHandle<int, FileCloserT> ScopedFile;
typedef SharedHandle<int, FileCloserT> SharedFile;

}

#endif
