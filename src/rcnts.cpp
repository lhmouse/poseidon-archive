// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#include "precompiled.hpp"
#include "rcnts.hpp"
#include "checked_arithmetic.hpp"
#include <memory>
#include <iostream>
#include <boost/make_shared.hpp>

namespace Poseidon {

namespace {
	template<typename T>
	class Incremental_alloc {
		template<typename>
		friend class Incremental_alloc;

	public:
		typedef T *             pointer;
		typedef const T *       const_pointer;
		typedef T &             reference;
		typedef const T &       const_reference;
		typedef T               value_type;
		typedef std::size_t     size_type;
		typedef std::ptrdiff_t  difference_type;

		template<typename U>
		struct rebind {
			typedef Incremental_alloc<U> other;
		};

	private:
		void **m_inc_ptr;
		size_type m_inc_size;

	public:
		Incremental_alloc(void **inc_ptr, size_type inc_size)
			: m_inc_ptr(inc_ptr), m_inc_size(inc_size)
		{
			//
		}
		template<typename U>
		Incremental_alloc(const Incremental_alloc<U> &rhs)
			: m_inc_ptr(rhs.m_inc_ptr), m_inc_size(rhs.m_inc_size)
		{
			//
		}

	public:
		pointer address(reference r) const {
			return reinterpret_cast<pointer>(&reinterpret_cast<char &>(r));
		}
		const_pointer address(const_reference r) const {
			return reinterpret_cast<const_pointer>(&reinterpret_cast<const char &>(r));
		}

		pointer allocate(size_type n, const void * = 0){
			size_type k = checked_mul(sizeof(T), n);
			if(m_inc_ptr){
				k = checked_add(k, m_inc_size);
			}
			void *const p = ::operator new(k);
			if(m_inc_ptr){
				*m_inc_ptr = static_cast<char *>(p) + k - m_inc_size;
			}
			return static_cast<pointer>(p);
		}
		void deallocate(pointer p, size_type){
			::operator delete(p);
		}
		size_type max_size() const {
			return static_cast<size_type>(-1) >> 1;
		}

		template<typename U>
		bool operator==(const Incremental_alloc<U> &) const {
			return true;
		}
		template<typename U>
		bool operator!=(const Incremental_alloc<U> &) const {
			return false;
		}

		void construct(pointer p, const T &t){
			new(static_cast<void *>(p)) T(t);
		}
#ifdef POSEIDON_CXX11
		void construct(pointer p, T &&t){
			new(static_cast<void *>(p)) T(std::move(t));
		}
#endif
		void destroy(pointer p){
			p->~T();
		}
	};
}

void Rcnts::assign(const char *str, std::size_t len){
	if(len == 0){
		m_ptr.reset(boost::shared_ptr<void>(), "");
	} else {
		void *dst;
		AUTO(sp, boost::allocate_shared<char>(Incremental_alloc<char>(&dst, len + 1)));
		std::memcpy(dst, str, len);
		static_cast<char *>(dst)[len] = 0;
		m_ptr.reset(STD_MOVE_IDN(sp), static_cast<const char *>(dst));
	}
}

std::istream &operator>>(std::istream &is, Rcnts &rhs){
	std::string str;
	if(is >>str){
		rhs.assign(str);
	}
	return is;
}
std::ostream &operator<<(std::ostream &os, const Rcnts &rhs){
	os <<rhs.get();
	return os;
}

}
