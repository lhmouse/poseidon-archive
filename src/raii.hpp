// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2017, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_RAII_HPP_
#define POSEIDON_RAII_HPP_

#include "cxx_ver.hpp"
#include <utility>
#include <ostream>

namespace Poseidon {

template<typename CloserT>
class UniqueHandle {
public:
	typedef VALUE_TYPE(DECLREF(CloserT)()) Handle;

private:
	Handle m_handle;

private:
	UniqueHandle(const UniqueHandle &);
	UniqueHandle &operator=(const UniqueHandle &);

public:
	UniqueHandle() NOEXCEPT
		: m_handle(CloserT()())
	{
	}
	explicit UniqueHandle(Handle rhs) NOEXCEPT
		: m_handle(rhs)
	{
	}
	UniqueHandle(Move<UniqueHandle> rhs) NOEXCEPT
		: m_handle(CloserT()())
	{
		rhs.swap(*this);
	}
	UniqueHandle &operator=(Move<UniqueHandle> rhs) NOEXCEPT {
		UniqueHandle(STD_MOVE(rhs)).swap(*this);
		return *this;
	}
	~UniqueHandle() NOEXCEPT {
		const Handle old = m_handle;
		if(old != CloserT()()){
			CloserT()(old);
		}
	}

public:
	Handle get() const NOEXCEPT {
		return m_handle;
	}
	Handle release() NOEXCEPT {
		const Handle ret = m_handle;
		m_handle = CloserT()();
		return ret;
	}
	UniqueHandle &reset(Handle rhs = CloserT()()) NOEXCEPT {
		UniqueHandle(rhs).swap(*this);
		return *this;
	}
	UniqueHandle &reset(Move<UniqueHandle> rhs) NOEXCEPT {
		UniqueHandle(STD_MOVE(rhs)).swap(*this);
		return *this;
	}

	void swap(UniqueHandle &rhs) NOEXCEPT {
		using std::swap;
		swap(m_handle, rhs.m_handle);
	}

public:
#ifdef POSEIDON_CXX11
	explicit operator bool() const noexcept {
		return get() != CloserT()();
	}
#else
	typedef Handle (UniqueHandle::*DummyBool_)() const;
	operator DummyBool_() const NOEXCEPT {
		return (get() != CloserT()()) ? &UniqueHandle::get : 0;
	}
#endif
};

template<typename CloserT>
void swap(UniqueHandle<CloserT> &lhs, UniqueHandle<CloserT> &rhs) NOEXCEPT {
	lhs.swap(rhs);
}

template<typename CloserT>
bool operator==(const UniqueHandle<CloserT> &lhs, const UniqueHandle<CloserT> &rhs){
	return lhs.get() == rhs.get();
}
template<typename CloserT>
bool operator==(const UniqueHandle<CloserT> &lhs, typename UniqueHandle<CloserT>::Handle rhs){
	return lhs.get() == rhs;
}
template<typename CloserT>
bool operator==(typename UniqueHandle<CloserT>::Handle lhs, const UniqueHandle<CloserT> &rhs){
	return lhs == rhs.get();
}

template<typename CloserT>
bool operator!=(const UniqueHandle<CloserT> &lhs, const UniqueHandle<CloserT> &rhs){
	return lhs.get() != rhs.get();
}
template<typename CloserT>
bool operator!=(const UniqueHandle<CloserT> &lhs, typename UniqueHandle<CloserT>::Handle rhs){
	return lhs.get() != rhs;
}
template<typename CloserT>
bool operator!=(typename UniqueHandle<CloserT>::Handle lhs, const UniqueHandle<CloserT> &rhs){
	return lhs != rhs.get();
}

template<typename CloserT>
bool operator<(const UniqueHandle<CloserT> &lhs, const UniqueHandle<CloserT> &rhs){
	return lhs.get() < rhs.get();
}
template<typename CloserT>
bool operator<(const UniqueHandle<CloserT> &lhs, typename UniqueHandle<CloserT>::Handle rhs){
	return lhs.get() < rhs;
}
template<typename CloserT>
bool operator<(typename UniqueHandle<CloserT>::Handle lhs, const UniqueHandle<CloserT> &rhs){
	return lhs < rhs.get();
}

template<typename CloserT>
bool operator>(const UniqueHandle<CloserT> &lhs, const UniqueHandle<CloserT> &rhs){
	return lhs.get() > rhs.get();
}
template<typename CloserT>
bool operator>(const UniqueHandle<CloserT> &lhs, typename UniqueHandle<CloserT>::Handle rhs){
	return lhs.get() > rhs;
}
template<typename CloserT>
bool operator>(typename UniqueHandle<CloserT>::Handle lhs, const UniqueHandle<CloserT> &rhs){
	return lhs > rhs.get();
}

template<typename CloserT>
bool operator<=(const UniqueHandle<CloserT> &lhs, const UniqueHandle<CloserT> &rhs){
	return lhs.get() <= rhs.get();
}
template<typename CloserT>
bool operator<=(const UniqueHandle<CloserT> &lhs, typename UniqueHandle<CloserT>::Handle rhs){
	return lhs.get() <= rhs;
}
template<typename CloserT>
bool operator<=(typename UniqueHandle<CloserT>::Handle lhs, const UniqueHandle<CloserT> &rhs){
	return lhs <= rhs.get();
}

template<typename CloserT>
bool operator>=(const UniqueHandle<CloserT> &lhs, const UniqueHandle<CloserT> &rhs){
	return lhs.get() >= rhs.get();
}
template<typename CloserT>
bool operator>=(const UniqueHandle<CloserT> &lhs, typename UniqueHandle<CloserT>::Handle rhs){
	return lhs.get() >= rhs;
}
template<typename CloserT>
bool operator>=(typename UniqueHandle<CloserT>::Handle lhs, const UniqueHandle<CloserT> &rhs){
	return lhs >= rhs.get();
}

template<typename CloserT>
std::ostream &operator<<(std::ostream &os, const UniqueHandle<CloserT> &handle){
	return os <<handle.get();
}

struct FileCloser {
	CONSTEXPR int operator()() const NOEXCEPT {
		return -1;
	}
	void operator()(int fd) const NOEXCEPT;
};

typedef UniqueHandle<FileCloser> UniqueFile;

}

#endif
