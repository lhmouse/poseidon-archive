#ifndef POSEIDON_ATOMIC_HPP_
#define POSEIDON_ATOMIC_HPP_

namespace Poseidon {

template<typename T>
struct AtomicIdentity_ {
	typedef T type;
};

template<typename T>
inline T atomicLoad(const volatile T &mem) throw() {
	volatile int barrier;
	__sync_lock_test_and_set(&barrier, 1);
	return mem;
}
template<typename T>
inline void atomicStore(volatile T &mem, typename AtomicIdentity_<T>::type val) throw() {
	mem = val;
	volatile int barrier;
	__sync_lock_release(&barrier);
}

inline void atomicSynchronize() throw() {
	__sync_synchronize();
}

template<typename T>
inline T atomicAdd(volatile T &mem, typename AtomicIdentity_<T>::type val) throw() {
	return __sync_add_and_fetch(&mem, val);
}
template<typename T>
inline T atomicSub(volatile T &mem, typename AtomicIdentity_<T>::type val) throw() {
	return __sync_sub_and_fetch(&mem, val);
}

template<typename T>
inline T atomicCmpExchange(volatile T &mem, typename AtomicIdentity_<T>::type cmp,
	typename AtomicIdentity_<T>::type xchg) throw()
{
	return __sync_val_compare_and_swap(&mem, cmp, xchg);
}
template<typename T>
inline T atomicExchange(volatile T &mem, typename AtomicIdentity_<T>::type xchg) throw() {
	T cmp = mem;
	for(;;){
		const T old = atomicCmpExchange(mem, cmp, xchg);
		if(old == cmp){
			break;
		}
		cmp = old;
	}
	return cmp;
}

}

#endif
