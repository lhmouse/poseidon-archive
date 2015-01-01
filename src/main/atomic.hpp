// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_ATOMIC_HPP_
#define POSEIDON_ATOMIC_HPP_

#include "cxx_ver.hpp"
#include "cxx_util.hpp"

#if __GNUC__ * 100 + __GNUC_MINOR__ >= 407
#	define GCC_HAS_ATOMIC_
#endif

namespace Poseidon {

enum MemModel {
#if __GNUC__ * 100 + __GNUC_MINOR__ >= 407
	ATOMIC_RELAXED = __ATOMIC_RELAXED,
	ATOMIC_ACQUIRE = __ATOMIC_ACQUIRE,
	ATOMIC_RELEASE = __ATOMIC_RELEASE,
	ATOMIC_ACQ_REL = __ATOMIC_ACQ_REL,
	ATOMIC_SEQ_CST = __ATOMIC_SEQ_CST,
#else
	ATOMIC_RELAXED = -1,
	ATOMIC_ACQUIRE = -1,
	ATOMIC_RELEASE = -1,
	ATOMIC_ACQ_REL = -1,
	ATOMIC_SEQ_CST = -1,
#endif
};

template<typename T>
inline T atomicLoad(const volatile T &mem, MemModel model) NOEXCEPT {
#ifdef GCC_HAS_ATOMIC_
	return __atomic_load_n(&mem, model);
#else
	(void)model;
	volatile int barrier;
	__sync_lock_test_and_set(&barrier, 1);
	return mem;
#endif
}
template<typename T>
inline void atomicStore(volatile T &mem, typename Identity<T>::type val, MemModel model) NOEXCEPT {
#ifdef GCC_HAS_ATOMIC_
	__atomic_store_n(&mem, val, model);
#else
	(void)model;
	mem = val;
	volatile int barrier;
	__sync_lock_release(&barrier);
#endif
}

inline void atomicFence(MemModel model) NOEXCEPT {
#ifdef GCC_HAS_ATOMIC_
	__atomic_thread_fence(model);
#else
	(void)model;
	__sync_synchronize();
#endif
}

template<typename T>
inline T atomicAdd(volatile T &mem, typename Identity<T>::type val, MemModel model) NOEXCEPT
{
#ifdef GCC_HAS_ATOMIC_
	return __atomic_add_fetch(&mem, val, model);
#else
	(void)model;
	return __sync_add_and_fetch(&mem, val);
#endif
}
template<typename T>
inline T atomicSub(volatile T &mem, typename Identity<T>::type val, MemModel model) NOEXCEPT {
#ifdef GCC_HAS_ATOMIC_
	return __atomic_sub_fetch(&mem, val, model);
#else
	(void)model;
	return __sync_sub_and_fetch(&mem, val);
#endif
}

template<typename T>
inline bool atomicCompareExchange(volatile T &mem, typename Identity<T>::type &cmp,
	typename Identity<T>::type xchg, MemModel modelSuccess, MemModel modelFailure) NOEXCEPT
{
#ifdef GCC_HAS_ATOMIC_
	return __atomic_compare_exchange_n(&mem, &cmp, xchg, false, modelSuccess, modelFailure);
#else
	(void)modelSuccess;
	(void)modelFailure;
	const T old = cmp;
	cmp = __sync_val_compare_and_swap(&mem, old, xchg);
	return cmp == old;
#endif
}
template<typename T>
inline T atomicExchange(volatile T &mem, typename Identity<T>::type xchg, MemModel model) NOEXCEPT {
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

}

#endif
