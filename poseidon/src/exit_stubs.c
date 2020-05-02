// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#include <stdlib.h>
#include <stdio.h>

void
exit(int status)
  {
    fputs(
      "WARNING: Please call `at_quick_exit()` to terminate the process.\n"
      "         Thorough cleanup is not feasible. We have to call `_Exit()` here.\n",
      stderr);

    fflush(0);
    _Exit(status);
  }

int
atexit(void (*func)(void))
  {
    (void)func;
    return 0;
  }
