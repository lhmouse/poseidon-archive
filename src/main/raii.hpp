#ifndef POSEIDON_RAII_HPP_
#define POSEIDON_RAII_HPP_

#include "../cxx_ver.hpp"
#include <utility>

namespace Poseidon {

template<typename CloserT>
class ScopedHandle {
public:
	typedef DECLTYPE(DECLREF(CloserT)()) Handle;

private:
	Handle m_handle;

public:
	ScopedHandle() NOEXCEPT
		: m_handle(CloserT()())
	{
	}
	explicit ScopedHandle(Handle handle) NOEXCEPT
		: m_handle(CloserT()())
	{
		reset(handle);
	}
	ScopedHandle(Move<ScopedHandle> rhs) NOEXCEPT
		: m_handle(CloserT()())
	{
		reset(STD_MOVE(rhs));
	}
	ScopedHandle &operator=(Handle handle) NOEXCEPT {
		reset(handle);
		return *this;
	}
	ScopedHandle &operator=(Move<ScopedHandle> rhs) NOEXCEPT {
		reset(STD_MOVE(rhs));
		return *this;
	}
	~ScopedHandle() NOEXCEPT {
		reset();
	}

private:
	ScopedHandle(const ScopedHandle &);
	void operator=(const ScopedHandle &);

public:
	Handle get() const NOEXCEPT {
		return m_handle;
	}
	Handle release() NOEXCEPT {
		const Handle ret = m_handle;
		m_handle = CloserT()();
		return ret;
	}
	void reset(Handle handle = CloserT()()) NOEXCEPT {
		const Handle old = m_handle;
		m_handle = handle;
		if(old != CloserT()()){
			CloserT()(old);
		}
	}
	void reset(Move<ScopedHandle> rhs) NOEXCEPT {
		rhs.swap(*this);
	}
	void swap(ScopedHandle &rhs) NOEXCEPT {
		using std::swap;
		swap(m_handle, rhs.m_handle);
	}
	ENABLE_IF_CXX11(explicit) operator bool() const NOEXCEPT {
		return get() != CloserT()();
	}
};

template<typename CloserT>
void swap(ScopedHandle<CloserT> &lhs, ScopedHandle<CloserT> &rhs) NOEXCEPT {
	lhs.swap(rhs);
}

#define DEFINE_RATIONAL_OPERATOR_(temp_, op_)	\
	template<typename CloserT>	\
	bool operator op_(const temp_<CloserT> &lhs,	\
		const temp_<CloserT> &rhs) NOEXCEPT	\
	{	\
		return lhs.get() op_ rhs.get();	\
	}	\
	template<typename CloserT>	\
	bool operator op_(typename temp_<CloserT>::Handle lhs,	\
		const temp_<CloserT> &rhs) NOEXCEPT	\
	{	\
		return lhs op_ rhs.get();	\
	}	\
	template<typename CloserT>	\
	bool operator op_(const temp_<CloserT> &lhs,	\
		typename temp_<CloserT>::Handle rhs) NOEXCEPT	\
	{	\
		return lhs.get() op_ rhs;	\
	}

DEFINE_RATIONAL_OPERATOR_(ScopedHandle, ==)
DEFINE_RATIONAL_OPERATOR_(ScopedHandle, !=)
DEFINE_RATIONAL_OPERATOR_(ScopedHandle, <)
DEFINE_RATIONAL_OPERATOR_(ScopedHandle, >)
DEFINE_RATIONAL_OPERATOR_(ScopedHandle, <=)
DEFINE_RATIONAL_OPERATOR_(ScopedHandle, >=)

#undef DEFINE_RATIONAL_OPERATOR_

extern void closeFile(int fd) NOEXCEPT;

struct FileCloser {
	int operator()() const NOEXCEPT {
		return -1;
	}
	void operator()(int fd) const NOEXCEPT {
		closeFile(fd);
	}
};

typedef ScopedHandle<FileCloser> ScopedFile;

}

#endif
