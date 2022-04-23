// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#include <stdio.h>

void
exit(int status);

int
atexit(void func(void));

int
__cxa_atexit(void func(void*), void* arg, void* dso_handle);

void
exit(int status)
  {
    // Exit immediately without any cleanups.
    fputs(
        "WARNING: Please call `quick_exit()` to terminate the process.\n"
        "         Thorough cleanup is not feasible.\n",
        stderr);

    fflush(0);
    _Exit(status);
  }

int
atexit(void func(void))
  {
    // Pretend that it has been registered successfully.
    (void)func;
    return 0;
  }

int
__cxa_atexit(void func(void*), void* arg, void* dso_handle)
  {
    // Pretend that it has been registered successfully.
    (void)func;
    (void)arg;
    (void)dso_handle;
    return 0;
  }
