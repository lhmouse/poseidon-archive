// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_RAII_HPP_
#define POSEIDON_RAII_HPP_

#include "cxx_ver.hpp"
#include <utility>
#include <ostream>

namespace Poseidon {

template<typename CloserT>
class Unique_handle {
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
	Unique_handle() NOEXCEPT
		: m_handle(CloserT()())
	{
		//
	}
	explicit Unique_handle(Handle rhs) NOEXCEPT
		: m_handle(rhs)
	{
		//
	}
	Unique_handle(Move<Unique_handle> rhs) NOEXCEPT
		: m_handle(CloserT()())
	{
		rhs.swap(*this);
	}
	Unique_handle &operator=(Move<Unique_handle> rhs) NOEXCEPT {
		Unique_handle(STD_MOVE(rhs)).swap(*this);
		return *this;
	}
	~Unique_handle() NOEXCEPT {
		const Handle old = m_handle;
		if(old != CloserT()()){
			CloserT()(old);
		}
	}
#ifdef POSEIDON_CXX11
	Unique_handle(const Unique_handle &) = delete;
	Unique_handle &operator=(const Unique_handle &) = delete;
#else
	// Move support.
	Unique_handle(Unique_handle &rhs) NOEXCEPT
		: m_handle(CloserT()())
	{
		rhs.swap(*this);
	}
	Unique_handle &operator=(Unique_handle &rhs) NOEXCEPT {
		Unique_handle(STD_MOVE(rhs)).swap(*this);
		return *this;
	}
	operator Move<Unique_handle> (){
		return Move<Unique_handle>(*this);
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
	Unique_handle &reset(Handle rhs = CloserT()()) NOEXCEPT {
		Unique_handle(rhs).swap(*this);
		return *this;
	}
	Unique_handle &reset(Move<Unique_handle> rhs) NOEXCEPT {
		Unique_handle(STD_MOVE(rhs)).swap(*this);
		return *this;
	}

	void swap(Unique_handle &rhs) NOEXCEPT {
		using std::swap;
		swap(m_handle, rhs.m_handle);
	}

public:
#ifdef POSEIDON_CXX11
	explicit operator bool() const noexcept {
		return get() != CloserT()();
	}
#else
	typedef Handle (Unique_handle::*Dummy_bool_)() const;
	operator Dummy_bool_() const NOEXCEPT {
		return (get() != CloserT()()) ? &Unique_handle::get : 0;
	}
#endif
};

template<typename CloserT>
void swap(Unique_handle<CloserT> &lhs, Unique_handle<CloserT> &rhs) NOEXCEPT {
	lhs.swap(rhs);
}

template<typename CloserT>
bool operator==(const Unique_handle<CloserT> &lhs, const Unique_handle<CloserT> &rhs){
	return lhs.get() == rhs.get();
}
template<typename CloserT>
bool operator==(const Unique_handle<CloserT> &lhs, typename Unique_handle<CloserT>::Handle rhs){
	return lhs.get() == rhs;
}
template<typename CloserT>
bool operator==(typename Unique_handle<CloserT>::Handle lhs, const Unique_handle<CloserT> &rhs){
	return lhs == rhs.get();
}

template<typename CloserT>
bool operator!=(const Unique_handle<CloserT> &lhs, const Unique_handle<CloserT> &rhs){
	return lhs.get() != rhs.get();
}
template<typename CloserT>
bool operator!=(const Unique_handle<CloserT> &lhs, typename Unique_handle<CloserT>::Handle rhs){
	return lhs.get() != rhs;
}
template<typename CloserT>
bool operator!=(typename Unique_handle<CloserT>::Handle lhs, const Unique_handle<CloserT> &rhs){
	return lhs != rhs.get();
}

template<typename CloserT>
bool operator<(const Unique_handle<CloserT> &lhs, const Unique_handle<CloserT> &rhs){
	return lhs.get() < rhs.get();
}
template<typename CloserT>
bool operator<(const Unique_handle<CloserT> &lhs, typename Unique_handle<CloserT>::Handle rhs){
	return lhs.get() < rhs;
}
template<typename CloserT>
bool operator<(typename Unique_handle<CloserT>::Handle lhs, const Unique_handle<CloserT> &rhs){
	return lhs < rhs.get();
}

template<typename CloserT>
bool operator>(const Unique_handle<CloserT> &lhs, const Unique_handle<CloserT> &rhs){
	return lhs.get() > rhs.get();
}
template<typename CloserT>
bool operator>(const Unique_handle<CloserT> &lhs, typename Unique_handle<CloserT>::Handle rhs){
	return lhs.get() > rhs;
}
template<typename CloserT>
bool operator>(typename Unique_handle<CloserT>::Handle lhs, const Unique_handle<CloserT> &rhs){
	return lhs > rhs.get();
}

template<typename CloserT>
bool operator<=(const Unique_handle<CloserT> &lhs, const Unique_handle<CloserT> &rhs){
	return lhs.get() <= rhs.get();
}
template<typename CloserT>
bool operator<=(const Unique_handle<CloserT> &lhs, typename Unique_handle<CloserT>::Handle rhs){
	return lhs.get() <= rhs;
}
template<typename CloserT>
bool operator<=(typename Unique_handle<CloserT>::Handle lhs, const Unique_handle<CloserT> &rhs){
	return lhs <= rhs.get();
}

template<typename CloserT>
bool operator>=(const Unique_handle<CloserT> &lhs, const Unique_handle<CloserT> &rhs){
	return lhs.get() >= rhs.get();
}
template<typename CloserT>
bool operator>=(const Unique_handle<CloserT> &lhs, typename Unique_handle<CloserT>::Handle rhs){
	return lhs.get() >= rhs;
}
template<typename CloserT>
bool operator>=(typename Unique_handle<CloserT>::Handle lhs, const Unique_handle<CloserT> &rhs){
	return lhs >= rhs.get();
}

template<typename CloserT>
std::ostream &operator<<(std::ostream &os, const Unique_handle<CloserT> &handle){
	return os <<handle.get();
}

struct File_closer {
	CONSTEXPR int operator()() const NOEXCEPT {
		return -1;
	}
	void operator()(int fd) const NOEXCEPT;
};

typedef Unique_handle<File_closer> Unique_file;

}

#endif
