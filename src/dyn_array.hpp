#ifndef POSEIDON_DYNARRAY_HPP_
#define POSEIDON_DYNARRAY_HPP_

#include "cxx_ver.hpp"
#include <new>
#include <stdexcept>
#include <iterator>
#include <cstddef>
#include <cassert>

namespace Poseidon {

template<typename ValueT>
class DynArray {
public:
	typedef ValueT				value_type;
	typedef const value_type &	const_reference;
	typedef value_type &		reference;
	typedef const value_type *	const_pointer;
	typedef value_type *		pointer;

	typedef std::size_t			size_type;
	typedef std::ptrdiff_t		difference_type;

	typedef const_pointer		const_iterator;
	typedef pointer				iterator;

	typedef std::reverse_iterator<const_pointer>	const_reverse_iterator;
	typedef std::reverse_iterator<iterator>			reverse_iterator;

private:
	pointer m_begin;
	pointer m_end;

private:
	DynArray(const DynArray &);
	DynArray &operator=(const DynArray &);

public:
#ifdef POSEIDON_CXX11
	template<typename ...ArgsT>
	explicit DynArray(size_type size, const ArgsT &...args){
		if(size == 0){
			m_begin = nullptr;
			m_end = nullptr;
		} else {
			const std::size_t sizeToAlloc = sizeof(value_type) * size;
			if(sizeToAlloc / sizeof(value_type) != size){
				throw std::bad_alloc();
			}
			m_begin = static_cast<pointer>(::operator new[](sizeToAlloc));
			m_end = m_begin;
			try {
				for(size_type i = 0; i < size; ++i){
					new(static_cast<void *>(m_end)) value_type(args...);
					++m_end;
				}
			} catch(...){
				while(m_end != m_begin){
					--m_end;
					m_end->~value_type();
				}
				throw;
			}
		}
	}
#else
	explicit DynArray(size_type size){
		if(size == 0){
			m_begin = nullptr;
			m_end = nullptr;
		} else {
			const std::size_t sizeToAlloc = sizeof(value_type) * size;
			if(sizeToAlloc / sizeof(value_type) != size){
				throw std::bad_array_new_length();
			}
			m_begin = ::operator new[](sizeToAlloc);
			m_end = m_begin;
			try {
				for(size_type i = 0; i < size; ++i){
					new(static_cast<void *>(m_end)) value_type();
					++m_end;
				}
			} catch(...){
				while(m_end != m_begin){
					--m_end;
					m_end->~value_type();
				}
				throw;
			}
		}
	}
	template<typename Arg0T>
	DynArray(size_type size, const Arg0T &arg0){
		if(size == 0){
			m_begin = nullptr;
			m_end = nullptr;
		} else {
			const std::size_t sizeToAlloc = sizeof(value_type) * size;
			if(sizeToAlloc / sizeof(value_type) != size){
				throw std::bad_array_new_length();
			}
			m_begin = ::operator new[](sizeToAlloc);
			m_end = m_begin;
			try {
				for(size_type i = 0; i < size; ++i){
					new(static_cast<void *>(m_end)) value_type(arg0);
					++m_end;
				}
			} catch(...){
				while(m_end != m_begin){
					--m_end;
					m_end->~value_type();
				}
				throw;
			}
		}
	}
	template<typename Arg0T, typename Arg1T>
	DynArray(size_type size, const Arg0T &arg1, const Arg0T &arg1){
		if(size == 0){
			m_begin = nullptr;
			m_end = nullptr;
		} else {
			const std::size_t sizeToAlloc = sizeof(value_type) * size;
			if(sizeToAlloc / sizeof(value_type) != size){
				throw std::bad_array_new_length();
			}
			m_begin = ::operator new[](sizeToAlloc);
			m_end = m_begin;
			try {
				for(size_type i = 0; i < size; ++i){
					new(static_cast<void *>(m_end)) value_type(arg0, arg1);
					++m_end;
				}
			} catch(...){
				while(m_end != m_begin){
					--m_end;
					m_end->~value_type();
				}
				throw;
			}
		}
	}
#endif
	~DynArray(){
		while(m_end != m_begin){
			--m_end;
			m_end->~value_type();
		}
		::operator delete[](m_begin);
	}

public:
	const_reference at(size_type idx) const {
		if(idx >= size()){
			throw std::out_of_range("Poseidon::DynArray::at");
		}
		return m_begin[idx];
	}
	reference at(size_type idx){
		if(idx >= size()){
			throw std::out_of_range("Poseidon::DynArray::at");
		}
		return m_begin[idx];
	}

	const_reference operator[](size_type idx) const NOEXCEPT {
		assert(idx < size());
		return m_begin[idx];
	}
	reference operator[](size_type idx) NOEXCEPT {
		assert(idx < size());
		return m_begin[idx];
	}

	const_reference front() const NOEXCEPT {
		assert(!empty());
		return m_begin[0];
	}
	reference front() NOEXCEPT {
		assert(!empty());
		return m_begin[0];
	}
	const_reference back() const NOEXCEPT {
		assert(!empty());
		return m_end[-1];
	}
	reference back() NOEXCEPT {
		assert(!empty());
		return m_end[-1];
	}

	const_iterator begin() const NOEXCEPT {
		return m_begin;
	}
	iterator begin() NOEXCEPT {
		return m_begin;
	}
	const_iterator cbegin() const NOEXCEPT {
		return m_begin;
	}
	const_iterator end() const NOEXCEPT {
		return m_end;
	}
	iterator end() NOEXCEPT {
		return m_end;
	}
	const_iterator cend() const NOEXCEPT {
		return m_end;
	}

	const_reverse_iterator rbegin() const NOEXCEPT {
		return const_reverse_iterator(m_begin);
	}
	reverse_iterator rbegin() NOEXCEPT {
		return reverse_iterator(m_begin);
	}
	const_reverse_iterator crbegin() const NOEXCEPT {
		return const_reverse_iterator(m_begin);
	}
	const_reverse_iterator rend() const NOEXCEPT {
		return const_reverse_iterator(m_end);
	}
	reverse_iterator rend() NOEXCEPT {
		return reverse_iterator(m_end);
	}
	const_reverse_iterator crend() const NOEXCEPT {
		return const_reverse_iterator(m_end);
	}

	bool empty() const NOEXCEPT {
		return m_begin == m_end;
	}
	size_type size() const NOEXCEPT {
		return static_cast<size_type>(m_end - m_begin);
	}
};

}

#endif
