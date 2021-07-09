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
    const uintptr_t m_key;

    atomic_relaxed<bool> m_zombie;
    atomic_relaxed<bool> m_resident;  // don't delete if orphaned
    atomic_relaxed<Async_State> m_state;

  protected:
    explicit
    Abstract_Async_Job(uintptr_t key) noexcept
      : m_key(key)
      { }

  protected:
    // Executes this job and satisfies some promise of the derived class.
    // This function is called only once. No matter whether it returns or
    // throws an exception, this job is deleted from the worker queue.
    virtual void
    do_execute()
      = 0;

  public:
    ASTERIA_NONCOPYABLE_DESTRUCTOR(Abstract_Async_Job);

    // Marks this job to be deleted immediately.
    bool
    shut_down() noexcept
      { return this->m_zombie.exchange(true);  }

    // Marks this job to be deleted if worker pool holds its last reference.
    bool
    set_resident(bool value = true) noexcept
      { return this->m_resident.exchange(value);  }

    // Gets the asynchrnous state, which is set by worker threads.
    ROCKET_PURE Async_State
    state() const noexcept
      { return this->m_state.load();  }
  };

}  // namespace poseidon

#endif
