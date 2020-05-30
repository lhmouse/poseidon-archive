// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_CORE_ABSTRACT_ASYNC_FUNCTION_HPP_
#define POSEIDON_CORE_ABSTRACT_ASYNC_FUNCTION_HPP_

#include "../fwd.hpp"

namespace poseidon {

class Abstract_Async_Function
  : public ::asteria::Rcfwd<Abstract_Async_Function>
  {
    friend Worker_Pool;

  private:
    uintptr_t m_key;
    ::std::atomic<bool> m_finished;

  public:
    explicit
    Abstract_Async_Function(uintptr_t key)
    noexcept
      : m_key(key), m_finished(false)
      { }

    ASTERIA_NONCOPYABLE_DESTRUCTOR(Abstract_Async_Function);

  protected:
    // Executes this function and stores a value somewhere.
    // This function is called only once. No matter whether it returns or
    // throws an exception, this object is deleted from the worker queue.
    virtual
    void
    do_execute()
      = 0;

    // Assigns an exception as the result.
    // This function is called after `do_execute()` throws an exception.
    // An overriden function should not throw exceptions. If another value
    // has already been assigned, this call shall have no effect.
    virtual
    void
    do_set_exception(const ::std::exception_ptr& eptr)
      = 0;

  public:
    // Checks whether a worker has finished this function.
    ROCKET_PURE_FUNCTION
    bool
    finished()
    const noexcept
      { return this->m_finished.load(::std::memory_order_acquire);  }
  };

}  // namespace poseidon

#endif
