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

    ::std::atomic<bool> m_resident;  // don't delete if orphaned
    ::std::atomic<Async_State> m_state;

  public:
    explicit
    Abstract_Async_Job(uintptr_t key)
    noexcept
      : m_key(key),
        m_resident(false), m_state(async_state_initial)
      { }

    ASTERIA_NONCOPYABLE_DESTRUCTOR(Abstract_Async_Job);

  private:
    void
    do_set_state(Async_State state)
    noexcept
      { this->m_state.store(state, ::std::memory_order_release);  }

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
    // Should this job be deleted if worker pool holds its last reference?
    ROCKET_PURE_FUNCTION
    bool
    resident()
    const noexcept
      { return this->m_resident.load(::std::memory_order_relaxed);  }

    void
    set_resident(bool value = true)
    noexcept
      { this->m_resident.store(value, ::std::memory_order_relaxed);  }

    // Gets the asynchrnous state, which is set by worker threads.
    ROCKET_PURE_FUNCTION
    Async_State
    state()
    const noexcept
      { return this->m_state.load(::std::memory_order_acquire);  }
  };

}  // namespace poseidon

#endif
