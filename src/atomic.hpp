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
inline T atomic_load(const volatile T &mem, MemOrder order) NOEXCEPT {
	return __atomic_load_n(&mem, order);
}
template<typename T>
inline void atomic_store(volatile T &mem, typename boost::common_type<T>::type val, MemOrder order) NOEXCEPT {
	__atomic_store_n(&mem, val, order);
}

inline void atomic_fence(MemOrder order) NOEXCEPT {
	__atomic_thread_fence(order);
}

template<typename T>
inline T atomic_add(volatile T &mem, typename boost::common_type<T>::type val, MemOrder order) NOEXCEPT {
	return __atomic_add_fetch(&mem, val, order);
}
template<typename T>
inline T atomic_sub(volatile T &mem, typename boost::common_type<T>::type val, MemOrder order) NOEXCEPT {
	return __atomic_sub_fetch(&mem, val, order);
}

template<typename T>
inline bool atomic_compare_exchange(volatile T &mem, typename boost::common_type<T>::type &cmp, typename boost::common_type<T>::type xchg, MemOrder order_success, MemOrder order_failure) NOEXCEPT {
	return __atomic_compare_exchange_n(&mem, &cmp, xchg, false, order_success, order_failure);
}
template<typename T>
inline T atomic_exchange(volatile T &mem, typename boost::common_type<T>::type xchg, MemOrder order) NOEXCEPT {
	return __atomic_exchange_n(&mem, xchg, order);
}

inline void atomic_pause() NOEXCEPT {
	__builtin_ia32_pause();
}

}

#endif
