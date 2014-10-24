#ifndef POSEIDON_ATOMIC_HPP_
#define POSEIDON_ATOMIC_HPP_

#include "cxx_util.hpp"

#if __GNUC__ * 100 + __GNUC_MINOR__ >= 407
#	define GCC_HAS_ATOMIC_
#endif

namespace Poseidon {

template<typename T>
inline T atomicLoad(const volatile T &mem) throw() {
#ifdef GCC_HAS_ATOMIC_
	return __atomic_load_n(&mem, __ATOMIC_SEQ_CST);
#else
	volatile int barrier;
	__sync_lock_test_and_set(&barrier, 1);
	return mem;
#endif
}
template<typename T>
inline void atomicStore(volatile T &mem, typename Identity<T>::type val) throw() {
#ifdef GCC_HAS_ATOMIC_
	__atomic_store_n(&mem, val, __ATOMIC_SEQ_CST);
#else
	mem = val;
	volatile int barrier;
	__sync_lock_release(&barrier);
#endif
}

inline void atomicSynchronize() throw() {
#ifdef GCC_HAS_ATOMIC_
	__atomic_thread_fence(__ATOMIC_SEQ_CST);
#else
	__sync_synchronize();
#endif
}

template<typename T>
inline T atomicAdd(volatile T &mem, typename Identity<T>::type val) throw() {
#ifdef GCC_HAS_ATOMIC_
	return __atomic_add_fetch(&mem, val, __ATOMIC_SEQ_CST);
#else
	return __sync_add_and_fetch(&mem, val);
#endif
}
template<typename T>
inline T atomicSub(volatile T &mem, typename Identity<T>::type val) throw() {
#ifdef GCC_HAS_ATOMIC_
	return __atomic_sub_fetch(&mem, val, __ATOMIC_SEQ_CST);
#else
	return __sync_sub_and_fetch(&mem, val);
#endif
}

template<typename T>
inline T atomicCmpExchange(volatile T &mem, typename Identity<T>::type cmp,
	typename Identity<T>::type xchg) throw()
{
#ifdef GCC_HAS_ATOMIC_
	__atomic_compare_exchange_n(&mem, &cmp, xchg, false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);
	return cmp;
#else
	return __sync_val_compare_and_swap(&mem, cmp, xchg);
#endif
}
template<typename T>
inline T atomicExchange(volatile T &mem, typename Identity<T>::type xchg) throw() {
#ifdef GCC_HAS_ATOMIC_
	return __atomic_exchange_n(&mem, xchg, __ATOMIC_SEQ_CST);
#else
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
