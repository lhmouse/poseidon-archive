// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_XUTILITIES_HPP_
#define POSEIDON_XUTILITIES_HPP_

#include "utilities.hpp"

namespace poseidon {

// Creates a thread that invokes `loopfnT` repeatedly and never exits.
// Exceptions thrown from the thread procedure are ignored.
template<void loopfnT(void*)>
::pthread_t
create_daemon_thread(const char* name, void* param = nullptr)
  {
    ROCKET_ASSERT_MSG(name, "no thread name specified");
    ROCKET_ASSERT_MSG(::std::strlen(name) <= 15, "thread name too long");

    // This is the thread routine. It never returns.
    struct Thunk
      {
        [[noreturn]] static
        void*
        do_loop(void* xparam)
          {
            // Disable cancellation for safety. Failure to set the cancel state is ignored.
            ::pthread_setcancelstate(PTHREAD_CANCEL_DISABLE, nullptr);

            // Execute `loopfnT` repeatedly. The thread never exits.
            do
              try {
                loopfnT(xparam);
              }
              catch(exception& stdex) {
                ::std::fprintf(stderr,
                  "WARNING: daemon error: %s\n"
                  "[exception `%s` thrown from %p (`%s`)]\n",
                  stdex.what(), typeid(stdex).name(),
                  reinterpret_cast<void*>(loopfnT), typeid(loopfnT).name());
              }
            while(true);
          }
      };

    // Create the thread first.
    ::pthread_t thr;
    int err = ::pthread_create(&thr, nullptr, Thunk::do_loop, param);
    if(err != 0)
      POSEIDON_THROW("could not create $2 thread\n"
                    "[`pthread_create()` failed: $1]",
                    format_errno(err), name);

    // Set the thread name. Failure to set the name is ignored.
    ::pthread_setname_np(thr, name);
    return thr;
  }

}  // namespace asteria

#endif
