// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#include <stdlib.h>
#include <stdio.h>

void
exit(int status)
  {
    fputs(
      "WARNING: Please call `quick_exit()` to terminate the process.\n"
      "         Thorough cleanup is not feasible. We have to call `_Exit()` here.\n",
      stderr);

    // We cannot call `quick_exit()`, which would violate the semantics
    // of `exit()`. However we cannot invoke callbacks that have been registered
    // by `atexit()` either, so just exit.
    fflush(0);
    _Exit(status);
  }

int
atexit(void (*func)(void))
  {
    // Pretend that it has been registered successfully.
    (void)func;
    return 0;
  }
