// This file is part of Poseidon.
// Copyleft 2022, LH_Mouse. All wrongs reserved.

#include "../precompiled.ipp"
#include "task_executor.hpp"
#include "async_logger.hpp"
#include "../core/abstract_task.hpp"
#include "../utils.hpp"

namespace poseidon {
namespace {

struct Queued_Task
  {
  };

}  // namespace

POSEIDON_HIDDEN_STRUCT(Task_Executor, Queued_Task);

Task_Executor::
Task_Executor()
  {
  }

Task_Executor::
~Task_Executor()
  {
  }

void
Task_Executor::
thread_loop()
  {
    ::sleep(1);
    POSEIDON_LOG_FATAL(("task executor"));
  }

}  // poseidon
