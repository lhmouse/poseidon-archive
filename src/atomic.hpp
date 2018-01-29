// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_ATOMIC_HPP_
#define POSEIDON_ATOMIC_HPP_

#include "cxx_ver.hpp"
#include "cxx_util.hpp"
#include <boost/type_traits/common_type.hpp>

namespace Poseidon {

enum MemOrder {
	memorder_relaxed = __ATOMIC_RELAXED,
	memorder_consume = __ATOMIC_CONSUME,
	memorder_acquire = __ATOMIC_ACQUIRE,
	memorder_release = __ATOMIC_RELEASE,
	memorder_acq_rel = __ATOMIC_ACQ_REL,
	memorder_seq_cst = __ATOMIC_SEQ_CST,
};

template<typename T>
inline T atomic_load(const volatile T &ref, MemOrder order) NOEXCEPT {
	return __atomic_load_n(&ref, order);
}
template<typename T, typename U>
inline void atomic_store(volatile T &ref, const U &val, MemOrder order) NOEXCEPT {
	__atomic_store_n(&ref, val, order);
}

inline void atomic_fence(MemOrder order) NOEXCEPT {
	__atomic_thread_fence(order);
}

template<typename T, typename U>
inline T atomic_add(volatile T &ref, const U &val, MemOrder order) NOEXCEPT {
	return __atomic_add_fetch(&ref, val, order);
}
template<typename T, typename U>
inline T atomic_sub(volatile T &ref, const U &val, MemOrder order) NOEXCEPT {
	return __atomic_sub_fetch(&ref, val, order);
}

template<typename T, typename U>
inline bool atomic_compare_exchange(volatile T &ref, T &cmp, const U &xchg, MemOrder order_success, MemOrder order_failure) NOEXCEPT {
	return __atomic_compare_exchange_n(&ref, &cmp, xchg, false, order_success, order_failure);
}
template<typename T, typename U>
inline T atomic_exchange(volatile T &ref, const U &xchg, MemOrder order) NOEXCEPT {
	return __atomic_exchange_n(&ref, xchg, order);
}

inline void atomic_pause() NOEXCEPT {
	__builtin_ia32_pause();
}

}

#endif
