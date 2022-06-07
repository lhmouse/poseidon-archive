// This file is part of Poseidon.
// Copyleft 2022, LH_Mouse. All wrongs reserved.

#include "precompiled.ipp"
#include "fwd.hpp"
#include "static/main_config.hpp"
#include <pthread.h>
#include <signal.h>
#include <string.h>
#include <stdio.h>

namespace poseidon {
namespace {

template<class ManagerT>
inline
ManagerT*
do_create_manager()
  {
    static ManagerT manager[1];

    // Return a pointer to the static instance.
    return manager;
  }

template<class ManagerT, typename ParamT = void>
inline
ManagerT*
do_create_manager_with_thread(const char* name, ParamT* param = nullptr)
  {
    static ManagerT manager[1];
    static ::pthread_t thrd_handle[1];
    static char thrd_name[16];

    auto thrd_function = [](void* thrd_param) -> void*
      {
        // Set thread information. Errors are ignored.
        ::sigset_t sigset;
        ::sigemptyset(&sigset);
        ::sigaddset(&sigset, SIGINT);
        ::sigaddset(&sigset, SIGTERM);
        ::sigaddset(&sigset, SIGHUP);
        ::sigaddset(&sigset, SIGALRM);
        ::pthread_sigmask(SIG_BLOCK, &sigset, nullptr);

        int oldst;
        ::pthread_setcancelstate(PTHREAD_CANCEL_DISABLE, &oldst);

        if(thrd_name[0] != 0)
          ::pthread_setname_np(::pthread_self(), thrd_name);

        // Enter an infinite loop.
        for(;;)
          try {
            manager->thread_loop((ParamT*) thrd_param);
          }
          catch(exception& stdex) {
            ::fprintf(stderr,
                "WARNING: Caught an exception from manager loop: %s\n"
                "[exception class `%s`]\n",
                stdex.what(), typeid(stdex).name());
          }
      };

    if(name)
      ::memcpy(thrd_name, name, ::std::min(::strlen(name), sizeof(thrd_name)));

    int err = ::pthread_create(thrd_handle, nullptr, thrd_function, (void*) param);
    if(err != 0)
      ::rocket::sprintf_and_throw<::std::runtime_error>(
          "Could not spawn manager thread: %s\n"
          "[`pthread_create()` failed: %d]",
          ::strerror(err), err);

    // Return a pointer to the static instance.
    return manager;
  }

}  // namespace

Main_Config* const main_config = do_create_manager<Main_Config>();

}  // namespace poseidon
