// This file is part of Poseidon.
// Copyleft 2022 - 2023, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_TEST_UTILS_
#define POSEIDON_TEST_UTILS_

#include "../poseidon/fwd.hpp"
#include "../poseidon/utils.hpp"
#include <unistd.h>   // ::alarm()

#define POSEIDON_TEST_CHECK(expr)  \
    do  \
      try {  \
        if(static_cast<bool>(expr) == false) {  \
          /* failed */  \
          ::asteria::write_log_to_stderr(__FILE__, __LINE__, __func__,  \
              ::rocket::sref("POSEIDON_TEST_CHECK FAIL: " #expr));  \
          \
          ::abort();  \
        }  \
        \
        /* successful */  \
        ::asteria::write_log_to_stderr(__FILE__, __LINE__, __func__,  \
            ::rocket::sref("POSEIDON_TEST_CHECK PASS: " #expr));  \
      }  \
      catch(::std::exception& stdex) {  \
        /* failed */  \
        ::asteria::write_log_to_stderr(__FILE__, __LINE__, __func__,  \
            ::rocket::cow_string("POSEIDON_TEST_CHECK EXCEPTION: " #expr)  \
              + "\n" + stdex.what());  \
        \
        ::abort();  \
      }  \
    while(false)

#define POSEIDON_TEST_CHECK_CATCH(expr)  \
    do  \
      try {  \
        (void) (expr);  \
        \
        /* failed */  \
        ::asteria::write_log_to_stderr(__FILE__, __LINE__, __func__,  \
            ::rocket::sref("POSEIDON_TEST_CHECK XPASS: " #expr));  \
        \
        ::abort();  \
      }  \
      catch(::std::exception& stdex) {  \
        /* successful */  \
        ::asteria::write_log_to_stderr(__FILE__, __LINE__, __func__,  \
            ::rocket::cow_string("POSEIDON_TEST_CHECK XFAIL: " #expr)  \
              + "\n" + stdex.what());  \
      }  \
    while(false)

// Set terminate handler.
static const auto poseidon_test_terminate = ::std::set_terminate(
    [] {
      auto eptr = ::std::current_exception();
      if(eptr) {
        try {
          ::std::rethrow_exception(eptr);
        }
        catch(::std::exception& stdex) {
          ::fprintf(stderr,
              "`::std::terminate()` called after `::std::exception`:\n%s\n",
              stdex.what());
        }
        catch(...) {
          ::fprintf(stderr,
              "`::std::terminate()` called after an unknown exception\n");
        }
      }
      else
        ::fprintf(stderr,
            "`::std::terminate()` called without an exception\n");

      ::fflush(nullptr);
      ::_Exit(1);
    });

// Set kill timer.
static const auto poseidon_test_alarm = ::alarm(30);

#endif
