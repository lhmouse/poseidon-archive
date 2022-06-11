// This file is part of Poseidon.
// Copyleft 2022, LH_Mouse. All wrongs reserved.

#include "precompiled.ipp"
#include "fwd.hpp"
#include "static/main_config.hpp"
#include "static/async_logger.hpp"
#include "static/timer_driver.hpp"
#include <pthread.h>
#include <signal.h>
#include <string.h>
#include <stdio.h>

namespace poseidon {
namespace {

template<class ManagerT>
inline
ManagerT&
do_create_manager()
  {
    static ManagerT manager;

    // Return a reference to the static instance.
    return manager;
  }

template<class ManagerT>
inline
ManagerT&
do_create_manager_with_thread(const char* name = nullptr, bool allow_signals = false)
  {
    static ManagerT manager;
    static ::pthread_t thrd_handle;
    static char thrd_name[16];
    static int thrd_signal_how = SIG_BLOCK;

    auto thrd_function = [](void*) -> void*
      {
        // Set thread information. Errors are ignored.
        int oldst;
        ::pthread_setcancelstate(PTHREAD_CANCEL_DISABLE, &oldst);

        if(thrd_name[0] != 0)
          ::pthread_setname_np(::pthread_self(), thrd_name);

        ::sigset_t sigset;
        ::sigemptyset(&sigset);
        ::sigaddset(&sigset, SIGINT);
        ::sigaddset(&sigset, SIGTERM);
        ::sigaddset(&sigset, SIGHUP);
        ::sigaddset(&sigset, SIGALRM);
        ::pthread_sigmask(thrd_signal_how, &sigset, nullptr);

        // Enter an infinite loop.
        for(;;)
          try {
            manager.thread_loop();
          }
          catch(exception& stdex) {
            ::fprintf(stderr,
                "WARNING: Caught an exception from manager loop: %s\n"
                "[exception class `%s`]\n",
                stdex.what(), typeid(stdex).name());
          }
      };

    if(name)
      ::memcpy(thrd_name, name, ::std::min(::strlen(name), sizeof(thrd_name) - 1));

    if(allow_signals)
      thrd_signal_how = SIG_UNBLOCK;

    // Create the thread. It is never joined or detached.
    int err = ::pthread_create(&thrd_handle, nullptr, thrd_function, nullptr);
    if(err != 0)
      ::rocket::sprintf_and_throw<::std::runtime_error>(
          "Could not spawn manager thread: %s\n"
          "[`pthread_create()` failed: %d]",
          ::strerror(err), err);

    // Return a reference to the static instance.
    return manager;
  }

}  // namespace

Main_Config& main_config = do_create_manager<Main_Config>();
Async_Logger& async_logger = do_create_manager_with_thread<Async_Logger>("logger");
Timer_Driver& timer_driver = do_create_manager_with_thread<Timer_Driver>("timer");
atomic_signal exit_signal;

}  // namespace poseidon
