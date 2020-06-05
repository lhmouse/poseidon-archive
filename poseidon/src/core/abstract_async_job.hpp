// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_CORE_ABSTRACT_ASYNC_JOB_HPP_
#define POSEIDON_CORE_ABSTRACT_ASYNC_JOB_HPP_

#include "../fwd.hpp"

namespace poseidon {

class Abstract_Async_Job
  : public ::asteria::Rcfwd<Abstract_Async_Job>
  {
    friend Worker_Pool;

  private:
    uintptr_t m_key;
    ::std::atomic<Async_State> m_state;

  public:
    explicit
    Abstract_Async_Job(uintptr_t key)
    noexcept
      : m_key(key), m_state(async_state_initial)
      { }

    ASTERIA_NONCOPYABLE_DESTRUCTOR(Abstract_Async_Job);

  protected:
    // Executes this job and satisfies some promise of the derived class.
    // This function is called only once. No matter whether it returns or
    // throws an exception, this job is deleted from the worker queue.
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
    // Gets the asynchrnous state, which is set by worker threads.
    ROCKET_PURE_FUNCTION
    Async_State
    state()
    const noexcept
      { return this->m_state.load(::std::memory_order_acquire);  }
  };

}  // namespace poseidon

#endif
