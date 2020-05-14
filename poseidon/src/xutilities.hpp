// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_XUTILITIES_HPP_
#define POSEIDON_XUTILITIES_HPP_

#include "utilities.hpp"

namespace poseidon {

// Creates a thread that invokes `funcT` repeatedly and never exits.
// If `false` is returned, the thread will sleep for a short period.
// Exceptions thrown from the thread procedure are ignored.
template<typename PointerT, bool funcT(PointerT)>
::pthread_t
create_daemon_thread(PointerT param, const char* name)
  {
    ROCKET_ASSERT_MSG(name, "no thread name specified");
    ROCKET_ASSERT_MSG(::std::strlen(name) < 15, "thread name too long");

    // This is the thread routine. It never returns.
    const auto thunk_proc = +[](void* ptr) -> void*
      {
        do
          try {
            bool busy = funcT(static_cast<PointerT>(ptr));
            if(!busy)
              ::usleep(10'000);
          }
          catch(::std::exception& stdex) {
            ::std::fprintf(stderr,
              "WARNING: daemon error: %s\n"
              "[exception `%s` thrown from %p (`%s`)]\n",
              stdex.what(), typeid(stdex).name(),
              reinterpret_cast<void*>(funcT), typeid(funcT).name());
          }
        while(true);
      };

    // Create the thread first.
    ::pthread_t thr;
    int err = ::pthread_create(&thr, nullptr, thunk_proc, param);
    if(err != 0)
      ASTERIA_THROW("could not create $2 thread\n"
                    "[`pthread_create()` failed: $1]'",
                    format_errno(err), name);

    // Set the thread name. Failure to set the name is ignored.
    ::pthread_setname_np(thr, name);

    return thr;
  }

}  // namespace asteria

#endif
