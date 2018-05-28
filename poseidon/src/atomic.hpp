// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_ATOMIC_HPP_
#define POSEIDON_ATOMIC_HPP_

#include "cxx_ver.hpp"
#include <boost/type_traits/common_type.hpp>

namespace Poseidon {

enum Memory_order {
	memory_order_relaxed = __ATOMIC_RELAXED,
	memory_order_consume = __ATOMIC_CONSUME,
	memory_order_acquire = __ATOMIC_ACQUIRE,
	memory_order_release = __ATOMIC_RELEASE,
	memory_order_acq_rel = __ATOMIC_ACQ_REL,
	memory_order_seq_cst = __ATOMIC_SEQ_CST,
};

template<typename RefT>
inline RefT atomic_load(const volatile RefT &ref, Memory_order order) NOEXCEPT {
	return __atomic_load_n(&ref, order);
}
template<typename RefT, typename ValueT>
inline void atomic_store(volatile RefT &ref, const ValueT &value, Memory_order order) NOEXCEPT {
	__atomic_store_n(&ref, value, order);
}

inline void atomic_fence(Memory_order order) NOEXCEPT {
	__atomic_thread_fence(order);
}

template<typename RefT, typename ValueT>
inline RefT atomic_add(volatile RefT &ref, const ValueT &value, Memory_order order) NOEXCEPT {
	return __atomic_add_fetch(&ref, value, order);
}
template<typename RefT, typename ValueT>
inline RefT atomic_sub(volatile RefT &ref, const ValueT &value, Memory_order order) NOEXCEPT {
	return __atomic_sub_fetch(&ref, value, order);
}

template<typename RefT, typename ValueT>
inline bool atomic_compare_exchange(volatile RefT &ref, RefT &cmp, const ValueT &xchg, Memory_order order_success, Memory_order order_failure) NOEXCEPT {
	return __atomic_compare_exchange_n(&ref, &cmp, xchg, false, order_success, order_failure);
}
template<typename RefT, typename ValueT>
inline RefT atomic_exchange(volatile RefT &ref, const ValueT &xchg, Memory_order order) NOEXCEPT {
	return __atomic_exchange_n(&ref, xchg, order);
}

inline void atomic_pause() NOEXCEPT {
	__builtin_ia32_pause();
}

}

#endif
