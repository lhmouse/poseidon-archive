// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#include <stdio.h>
#include <stdlib.h>
#pragma GCC diagnostic ignored "-Wmissing-declarations"

using atexit_callback = void (void);
using cxa_atexit_callback = void (void*);

void
exit(int status)
  {
    // Exit immediately without any cleanups.
    ::fputs("WARNING: Please call `quick_exit()` instead.\n", stderr);
    ::fflush(nullptr);
    ::_Exit(status);
  }

int
atexit(atexit_callback* func)
  {
    // Pretend that it has been registered successfully.
    (void) func;
    return 0;
  }

int
__cxa_atexit(cxa_atexit_callback* dtor, void* obj, void* dso)
  {
    // Pretend that it has been registered successfully.
    (void) dtor;
    (void) obj;
    (void) dso;
    return 0;
  }
