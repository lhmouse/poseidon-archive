// This file is part of Poseidon.
// Copyleft 2022, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_TASK_ABSTRACT_TASK_
#define POSEIDON_TASK_ABSTRACT_TASK_

#include "../fwd.hpp"

namespace poseidon {

class Abstract_Async_Task
  {
  private:
    friend class Async_Task_Executor;

    atomic_relaxed<Async_State> m_state = { async_state_pending };

  protected:
    // Constructs an asynchronous task.
    explicit
    Abstract_Async_Task() noexcept;

  protected:
    // This callback is invoked by the task executor thread and is intended to
    // be overriden by derived classes.
    virtual
    void
    do_abstract_task_on_execution() = 0;

  public:
    ASTERIA_NONCOPYABLE_VIRTUAL_DESTRUCTOR(Abstract_Async_Task);

    // Gets the schedule state.
    Async_State
    async_state() const noexcept
      { return this->m_state.load();  }
  };

}  // namespace poseidon

#endif
