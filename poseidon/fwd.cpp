// This file is part of Poseidon.
// Copyleft 2022, LH_Mouse. All wrongs reserved.

#include "precompiled.ipp"
#include "fwd.hpp"
#include "static/main_config.hpp"
#include "static/async_logger.hpp"
#include "static/timer_driver.hpp"
#include "static/fiber_scheduler.hpp"
#include "static/async_task_executor.hpp"
#include "static/network_driver.hpp"
#include <pthread.h>
#include <signal.h>
#include <string.h>
#include <stdio.h>

namespace poseidon {
namespace {

template<class MgrT>
MgrT&
operator|(MgrT& mgr, const char* thrd_name)
  {
    auto thrd_function = +[](void* ptr) noexcept
      {
        // Set thread information. Errors are ignored.
        int oldst;
        ::pthread_setcancelstate(PTHREAD_CANCEL_DISABLE, &oldst);

        ::sigset_t sigset;
        ::sigemptyset(&sigset);
        ::sigaddset(&sigset, SIGINT);
        ::sigaddset(&sigset, SIGTERM);
        ::sigaddset(&sigset, SIGHUP);
        ::sigaddset(&sigset, SIGALRM);
        ::pthread_sigmask(SIG_BLOCK, &sigset, nullptr);

        // Enter an infinite loop.
        for(;;)
          try {
            ((MgrT*) ptr)->thread_loop();
          }
          catch(exception& stdex) {
            ::fprintf(stderr,
                "WARNING: Caught an exception from manager loop: %s\n"
                "[manager class `%s`]\n"
                "[exception class `%s`]\n",
                stdex.what(), typeid(MgrT).name(), typeid(stdex).name());
          }

        // Make the return type deducible.
        return (void*) nullptr;
      };

    // Create a detached thread.
    ::pthread_t thrd;
    int err = ::pthread_create(&thrd, nullptr, thrd_function, ::std::addressof(mgr));
    if(err != 0)
      ::rocket::sprintf_and_throw<::std::runtime_error>(
          "Could not spawn manager thread: %s\n"
          "[`pthread_create()` failed: %d]"
          "[manager class `%s`]\n",
          ::strerror(err), err, typeid(MgrT).name());

    // Name the thread and detach it. Errors are ignored.
    ::pthread_setname_np(thrd, thrd_name);
    ::pthread_detach(thrd);
    return mgr;
  }

}  // namespace

atomic_signal exit_signal;
Main_Config& main_config = *new Main_Config;
Fiber_Scheduler& fiber_scheduler = *new Fiber_Scheduler;

Async_Logger& async_logger = *new Async_Logger | "logger";
Timer_Driver& timer_driver = *new Timer_Driver | "timer";
Async_Task_Executor& async_task_executor = *new Async_Task_Executor | "task1" | "task2" | "task3";
Network_Driver& network_driver = *new Network_Driver | "network";

}  // namespace poseidon
