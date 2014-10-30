#include "cxx_ver.hpp"
#include "shared_ntmbs.hpp"
#include "log.hpp"
#include <memory>
#include <iostream>
#include <boost/weak_ptr.hpp>
#include <boost/make_shared.hpp>
using namespace Poseidon;

namespace {

template<typename T>
struct IncrementalAlloc {
	typedef T *				pointer;
	typedef const T *		const_pointer;
	typedef T &				reference;
	typedef const T &		const_reference;
	typedef T				value_type;
	typedef std::size_t		size_type;
	typedef std::ptrdiff_t	difference_type;

	template<typename U>
	struct rebind {
		typedef IncrementalAlloc<U> other;
	};

	void *&incPtr;
	const size_type incSize;

	IncrementalAlloc(void *&incPtr_, size_type incSize_)
		: incPtr(incPtr_), incSize(incSize_)
	{
	}
	template<typename U>
	IncrementalAlloc(const IncrementalAlloc<U> &rhs)
		: incPtr(rhs.incPtr), incSize(rhs.incSize)
	{
	}

	pointer address(reference r) const {
		return reinterpret_cast<pointer>(&reinterpret_cast<char &>(r));
	}
	const_pointer address(const_reference r) const {
		return reinterpret_cast<const_pointer>(&reinterpret_cast<const char &>(r));
	}

	pointer allocate(size_type n, const void * = 0){
		const size_type k = n * sizeof(T) + incSize;
		if(k / sizeof(T) != n + incSize / sizeof(T)){
			throw std::bad_alloc();
		}
		char *const ptr = reinterpret_cast<char *>(::operator new(k));
		incPtr = ptr + k - incSize;
		return reinterpret_cast<pointer>(ptr);
	}
	void deallocate(pointer p, size_type){
		::operator delete(p);
	}
	size_type max_size() const {
		return static_cast<size_type>(-1) >> 1;
	}

	template<typename U>
	bool operator==(const IncrementalAlloc<U> &) const {
		return true;
	}
	template<typename U>
	bool operator!=(const IncrementalAlloc<U> &) const {
		return false;
	}

	void construct(pointer p, const T &t){
		new(static_cast<void *>(p)) T(t);
	}
#ifdef POSEIDON_CXX11
	void construct(pointer p, T &&t){
		new(static_cast<void *>(p)) T(static_cast<T &&>(t));
	}
#endif
	void destroy(pointer p){
		p->~T();
	}
};

const boost::weak_ptr<const char> NULL_WEAK_PTR;

}

bool SharedNtmbs::isOwning() const {
	// FIXME: 低版本 boost 没有 owner_less。
	const boost::weak_ptr<const char> ownerTest(m_ptr);
	return (ownerTest < NULL_WEAK_PTR) || (NULL_WEAK_PTR < ownerTest);
}
void SharedNtmbs::forkOwning(){
	if(!isOwning()){
		void *dst;
		const std::size_t size = std::strlen(m_ptr.get()) + 1;
		const AUTO(sp, boost::allocate_shared<char>(IncrementalAlloc<char>(dst, size)));
		std::memcpy(dst, m_ptr.get(), size);
		m_ptr.reset(sp, static_cast<const char *>(dst));
	}
}

namespace Poseidon {

std::ostream &operator<<(std::ostream &os, const SharedNtmbs &rhs){
	return os <<rhs.get();
}

}
