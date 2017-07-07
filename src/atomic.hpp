// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2017, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_ATOMIC_HPP_
#define POSEIDON_ATOMIC_HPP_

#include "cxx_ver.hpp"
#include "cxx_util.hpp"
#include <boost/type_traits/common_type.hpp>

#if __GNUC__ * 100 + __GNUC_MINOR__ >= 407
#   define GCC_HAS_ATOMIC_ 1
#endif

namespace Poseidon {

enum MemModel {
#ifdef GCC_HAS_ATOMIC_
	ATOMIC_RELAXED = __ATOMIC_RELAXED,
	ATOMIC_CONSUME = __ATOMIC_CONSUME,
	ATOMIC_ACQUIRE = __ATOMIC_ACQUIRE,
	ATOMIC_RELEASE = __ATOMIC_RELEASE,
	ATOMIC_ACQ_REL = __ATOMIC_ACQ_REL,
	ATOMIC_SEQ_CST = __ATOMIC_SEQ_CST,
#else
	ATOMIC_RELAXED = -1,
	ATOMIC_CONSUME = -1,
	ATOMIC_ACQUIRE = -1,
	ATOMIC_RELEASE = -1,
	ATOMIC_ACQ_REL = -1,
	ATOMIC_SEQ_CST = -1,
#endif
};

template<typename T>
inline T atomic_load(const volatile T &mem, MemModel model) NOEXCEPT {
#ifdef GCC_HAS_ATOMIC_
	return __atomic_load_n(&mem, model);
#else
	(void)model;
	T val = mem;
	__sync_synchronize();
	return val;
#endif
}
template<typename T>
inline void atomic_store(volatile T &mem, typename boost::common_type<T>::type val, MemModel model) NOEXCEPT {
#ifdef GCC_HAS_ATOMIC_
	__atomic_store_n(&mem, val, model);
#else
	(void)model;
	__sync_synchronize();
	mem = val;
#endif
}

inline void atomic_fence(MemModel model) NOEXCEPT {
#ifdef GCC_HAS_ATOMIC_
	__atomic_thread_fence(model);
#else
	(void)model;
	__sync_synchronize();
#endif
}

template<typename T>
inline T atomic_add(volatile T &mem, typename boost::common_type<T>::type val, MemModel model) NOEXCEPT {
#ifdef GCC_HAS_ATOMIC_
	return __atomic_add_fetch(&mem, val, model);
#else
	(void)model;
	return __sync_add_and_fetch(&mem, val);
#endif
}
template<typename T>
inline T atomic_sub(volatile T &mem, typename boost::common_type<T>::type val, MemModel model) NOEXCEPT {
#ifdef GCC_HAS_ATOMIC_
	return __atomic_sub_fetch(&mem, val, model);
#else
	(void)model;
	return __sync_sub_and_fetch(&mem, val);
#endif
}

template<typename T>
inline bool atomic_compare_exchange(volatile T &mem, typename boost::common_type<T>::type &cmp,
	typename boost::common_type<T>::type xchg, MemModel model_success, MemModel model_failure) NOEXCEPT
{
#ifdef GCC_HAS_ATOMIC_
	return __atomic_compare_exchange_n(&mem, &cmp, xchg, false, model_success, model_failure);
#else
	(void)model_success;
	(void)model_failure;
	const T old = cmp;
	cmp = __sync_val_compare_and_swap(&mem, old, xchg);
	return cmp == old;
#endif
}
template<typename T>
inline T atomic_exchange(volatile T &mem, typename boost::common_type<T>::type xchg, MemModel model) NOEXCEPT {
#ifdef GCC_HAS_ATOMIC_
	return __atomic_exchange_n(&mem, xchg, model);
#else
	(void)model;
	T cmp = mem;
	for(;;){
		const T old = __sync_val_compare_and_swap(&mem, cmp, xchg);
		if(old == cmp){
			break;
		}
		cmp = old;
	}
	return cmp;
#endif
}

inline void atomic_pause() NOEXCEPT {
	__builtin_ia32_pause();
}

}

#endif
