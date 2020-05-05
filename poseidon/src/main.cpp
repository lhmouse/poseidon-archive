// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#include "precompiled.hpp"
#include <locale.h>
#include <unistd.h>
#include <signal.h>
#include <stdlib.h>
#include <stdio.h>

//using namespace poseidon;

namespace {

[[noreturn]]
int
do_print_help_and_exit(const char* self)
  {
    ::printf(
//        1         2         3         4         5         6         7     |
// 3456789012345678901234567890123456789012345678901234567890123456789012345|
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""" R"'''''''''''''''(
Usage: %s [OPTIONS] [[--] FILE [ARGUMENTS]...]

  -h      show help message then exit
  -V      show version information then exit

Visit the homepage at <%s>.
Report bugs to <%s>.
)'''''''''''''''" """"""""""""""""""""""""""""""""""""""""""""""""""""""""+1,
// 3456789012345678901234567890123456789012345678901234567890123456789012345|
//        1         2         3         4         5         6         7     |
      self,
      PACKAGE_URL,
      PACKAGE_BUGREPORT);

    ::exit(0);
  }

const char*
do_tell_build_time()
  {
    static char s_time_str[64];
    if(ROCKET_EXPECT(s_time_str[0]))
      return s_time_str;

    // Convert the build time to ISO 8601 format.
    ::tm tr;
    ::std::memset(&tr, 0, sizeof(tr));
    ::strptime(__DATE__ " " __TIME__, "%b %d %Y %H:%M:%S", &tr);
    ::strftime(s_time_str, sizeof(s_time_str), "%Y-%m-%d %H:%M:%S", &tr);
    return s_time_str;
  }

[[noreturn]]
int
do_print_version_and_exit()
  {
    ::printf(
//        1         2         3         4         5         6         7     |
// 3456789012345678901234567890123456789012345678901234567890123456789012345|
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""" R"'''''''''''''''(
%s [built on %s]

Visit the homepage at <%s>.
Report bugs to <%s>.
)'''''''''''''''" """"""""""""""""""""""""""""""""""""""""""""""""""""""""+1,
// 3456789012345678901234567890123456789012345678901234567890123456789012345|
//        1         2         3         4         5         6         7     |
      PACKAGE_STRING, do_tell_build_time(),
      PACKAGE_URL,
      PACKAGE_BUGREPORT);

    ::exit(0);
  }

// We want to detect Ctrl-C.
volatile ::sig_atomic_t interrupted;

void
do_trap_sigint()
noexcept
  {
    // Trap Ctrl-C. Failure to set the signal handler is ignored.
    struct ::sigaction sigx = { };
    sigx.sa_handler = [](int) { interrupted = 1;  };
    ::sigaction(SIGINT, &sigx, nullptr);
  }

void
do_parse_command_line(int argc, char** argv)
  {
    bool help = false;
    bool version = false;

    // Check for some common options before calling `getopt()`.
    if(argc > 1) {
      if(::strcmp(argv[1], "--help") == 0)
        do_print_help_and_exit(argv[0]);

      if(::strcmp(argv[1], "--version") == 0)
        do_print_version_and_exit();
    }

    // Parse command-line options.
    int ch;
    while((ch = ::getopt(argc, argv, "+hV")) != -1) {
      // Identify a single option.
      switch(ch) {
        case 'h':
          help = true;
          continue;

        case 'V':
          version = true;
          continue;
      }

      // `getopt()` will have written an error message to standard error.
      ::fprintf(stderr, "Try `%s -h` for help.\n", argv[0]);
      ::exit(2);
    }

    // Check for early exit conditions.
    if(help)
      do_print_help_and_exit(argv[0]);

    if(version)
      do_print_version_and_exit();

    // If more arguments follow, they denote the script to execute.
    if(optind < argc) {
// TODO
    }
  }

}  // namespace

int
main(int argc, char** argv)
  {
    // Select the C locale. UTF-8 is required for wide-oriented standard streams.
    ::setlocale(LC_ALL, "C.UTF-8");

    // Note that this function shall not return in case of errors.
    do_parse_command_line(argc, argv);

  }
