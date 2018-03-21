// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_RAII_HPP_
#define POSEIDON_RAII_HPP_

#include "cxx_ver.hpp"
#include <utility>
#include <ostream>

namespace Poseidon {

template<typename CloserT>
class UniqueHandle {
private:
	template<typename T>
	static T decay_helper(const T &) NOEXCEPT;

public:
#ifdef POSEIDON_CXX11
	using Handle = decltype(decay_helper(std::declval<CloserT>()()));
#else
	typedef __typeof__(decay_helper(DECLREF(CloserT)())) Handle;
#endif

private:
	Handle m_handle;

public:
	UniqueHandle() NOEXCEPT
		: m_handle(CloserT()())
	{
		//
	}
	explicit UniqueHandle(Handle rhs) NOEXCEPT
		: m_handle(rhs)
	{
		//
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
#ifdef POSEIDON_CXX11
	UniqueHandle(const UniqueHandle &) = delete;
	UniqueHandle &operator=(const UniqueHandle &) = delete;
#else
	// Move support.
	UniqueHandle(UniqueHandle &rhs) NOEXCEPT
		: m_handle(CloserT()())
	{
		rhs.swap(*this);
	}
	UniqueHandle &operator=(UniqueHandle &rhs) NOEXCEPT {
		UniqueHandle(STD_MOVE(rhs)).swap(*this);
		return *this;
	}
	operator Move<UniqueHandle> (){
		return Move<UniqueHandle>(*this);
	}
#endif

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
