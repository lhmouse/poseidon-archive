// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#include "precompiled.hpp"
#include "core/config_file.hpp"
#include "static/main_config.hpp"
#include "static/async_logger.hpp"
#include "static/timer_driver.hpp"
#include "static/network_driver.hpp"
#include "static/worker_pool.hpp"
#include "static/fiber_scheduler.hpp"
#include "utilities.hpp"
#include <locale.h>
#include <unistd.h>
#include <signal.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>
#include <dlfcn.h>

using namespace poseidon;

namespace {

[[noreturn]]
int
do_print_help_and_exit(const char* self)
  {
    ::printf(
//        1         2         3         4         5         6         7     |
// 3456789012345678901234567890123456789012345678901234567890123456789012345|
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""" R"'''''''''''''''(
Usage: %s [OPTIONS] [[--] DIRECTORY]

  -d      daemonize
  -h      show help message then exit
  -V      show version information then exit
  -v      enable verbose mode

Daemonization, if requested, is performed after loading config files. Early
failues are printed to standard error.

If DIRECTORY is specified, the working directory is switched there before
doing everything else.

Visit the homepage at <%s>.
Report bugs to <%s>.
)'''''''''''''''" """"""""""""""""""""""""""""""""""""""""""""""""""""""""+1,
// 3456789012345678901234567890123456789012345678901234567890123456789012345|
//        1         2         3         4         5         6         7     |
      self,
      PACKAGE_URL,
      PACKAGE_BUGREPORT);

    ::fflush(stdout);
    ::quick_exit(0);
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

    ::fflush(stdout);
    ::quick_exit(0);
  }

// We want to detect Ctrl-C.
::std::atomic<int> exit_sig;

void
do_set_exit_sig(int sig)
  {
    exit_sig.store(sig, ::std::memory_order_relaxed);
  }

void
do_trap_exit_signal(int sig)
noexcept
  {
    // Trap Ctrl-C. Failure to set the signal handler is ignored.
    struct ::sigaction sigx = { };
    sigx.sa_handler = do_set_exit_sig;
    ::sigaction(sig, &sigx, nullptr);
  }

// Define command-line options here.
struct Command_Line_Options
  {
    // options
    bool daemonize = false;
    bool verbose = false;

    // non-options
    cow_string cd_here;
  };

// This may also be automatic objects. It is declared here for convenience.
Command_Line_Options cmdline;

// These are process exit status codes.
enum Exit_Code : uint8_t
  {
    exit_success            = 0,
    exit_system_error       = 1,
    exit_invalid_argument   = 2,
  };

[[noreturn]]
int
do_exit(Exit_Code code, const char* fmt = nullptr, ...)
noexcept
  {
    // Sleep for a few moments so pending logs are flushed.
    if(Async_Logger::queue_size() != 0) {
      ::usleep(100'000);

      if(Async_Logger::queue_size() != 0)
        ::sleep(5);
    }

    // Output the string to standard error.
    if(fmt) {
      ::va_list ap;
      va_start(ap, fmt);
      ::vfprintf(stderr, fmt, ap);
      va_end(ap);
    }

    // Perform fast exit.
    ::fflush(nullptr);
    ::quick_exit(static_cast<int>(code));
  }

void
do_parse_command_line(int argc, char** argv)
  {
    bool help = false;
    bool version = false;

    opt<bool> daemonize;
    opt<bool> verbose;
    opt<cow_string> cd_here;

    // Check for some common options before calling `getopt()`.
    if(argc > 1) {
      if(::strcmp(argv[1], "--help") == 0)
        do_print_help_and_exit(argv[0]);

      if(::strcmp(argv[1], "--version") == 0)
        do_print_version_and_exit();
    }

    // Parse command-line options.
    int ch;
    while((ch = ::getopt(argc, argv, "+dhVv")) != -1) {
      // Identify a single option.
      switch(ch) {
        case 'd':
          daemonize = true;
          continue;

        case 'h':
          help = true;
          continue;

        case 'V':
          version = true;
          continue;

        case 'v':
          verbose = true;
          continue;
      }

      // `getopt()` will have written an error message to standard error.
      do_exit(exit_invalid_argument, "Try `%s -h` for help.\n",
                                     argv[0]);
    }

    // Check for early exit conditions.
    if(help)
      do_print_help_and_exit(argv[0]);

    if(version)
      do_print_version_and_exit();

    // If more arguments follow, they denote the working directory.
    if(argc - optind > 1)
      do_exit(exit_invalid_argument, "%s: too many arguments -- '%s'\n"
                                     "Try `%s -h` for help.\n",
                                     argv[0], argv[optind+1],
                                     argv[0]);

    if(argc - optind > 0)
      cd_here = cow_string(argv[optind]);

    // Daemonization mode is off by default.
    if(daemonize)
      cmdline.daemonize = *daemonize;

    // Verbose mode is off by default.
    if(verbose)
      cmdline.verbose = *verbose;

    // The default working directory is empty which means 'do not switch'.
    if(cd_here)
      cmdline.cd_here = ::std::move(*cd_here);
  }

::std::deque<cow_string>
do_get_addons()
  {
    auto file = Main_Config::copy();

    ::std::deque<cow_string> addons;
    auto qarr = file.get_array_opt({"addons"});
    if(!qarr)
      return addons;

    POSEIDON_LOG_DEBUG("Parsed list of add-ons: $1", *qarr);
    for(const auto& val : *qarr) {
      if(!val.is_string())
        POSEIDON_THROW("Invalid add-on path (`$1` is not a string)", val);
      addons.emplace_back(val.as_string());
    }
    return addons;
  }

}  // namespace

int
main(int argc, char** argv)
  try {
    // Select the C locale.
    // UTF-8 is required for wide-oriented standard streams.
    ::setlocale(LC_ALL, "C.UTF-8");

    // Note that this function shall not return in case of errors.
    do_parse_command_line(argc, argv);

    // Set current working directory if one is specified.
    if(cmdline.cd_here.size())
      if(::chdir(cmdline.cd_here.safe_c_str()) != 0)
        POSEIDON_THROW("Could not set working directory to '$2'\n"
                       "[`chdir()` failed: $1]",
                       noadl::format_errno(errno), cmdline.cd_here);

    // Load 'main.conf' before daemonization, so any earlier failures are
    // visible to the user.
    Main_Config::reload();
    Async_Logger::reload();
    Network_Driver::reload();
    Worker_Pool::reload();
    Fiber_Scheduler::reload();

    // Daemonize the process before entering modal loop.
    if(cmdline.daemonize)
      if(::daemon(1, 0) != 0)
        POSEIDON_THROW("Could not daemonize process\n"
                       "[`chdir()` failed: $1]",
                       noadl::format_errno(errno));

    const auto pid = ::getpid();

    // Set name of the main thread. Failure to set the name is ignored.
    ::pthread_setname_np(::pthread_self(), "poseidon");

    // Disable cancellation for safety. Failure to set the cancel state is ignored.
    ::pthread_setcancelstate(PTHREAD_CANCEL_DISABLE, nullptr);

    // Trap exit signals. Failure to set signal handlers is ignored.
    // This also makes stdio functions fail immediately.
    do_trap_exit_signal(SIGINT);
    do_trap_exit_signal(SIGTERM);
    do_trap_exit_signal(SIGHUP);

    // Ignore `SIGPIPE` for good.
    ::signal(SIGPIPE, SIG_IGN);

    // Start daemon threads and load add-ons.
    POSEIDON_LOG_INFO("Starting up: $1 (PID $2)", PACKAGE_STRING, pid);

    Async_Logger::start();
    Timer_Driver::start();
    Network_Driver::start();

    for(const auto& path : do_get_addons()) {
      POSEIDON_LOG_INFO("Loading add-on: $1", path);
      if(::dlopen(path.safe_c_str(), RTLD_NOW | RTLD_LOCAL | RTLD_NODELETE) == nullptr)
        POSEIDON_THROW("Error loading add-on '$1'\n"
                       "[`dlopen()` failed: $2]",
                       path, ::dlerror());
      POSEIDON_LOG_INFO("Finished loading add-on: $1", path);
    }

    POSEIDON_LOG_INFO("Started up and running: $1 (PID $2)", PACKAGE_STRING, pid);

    // Schedule fibers until a termination signal is caught.
    Fiber_Scheduler::modal_loop(exit_sig);
    const auto sig = exit_sig.load(::std::memory_order_relaxed);
    POSEIDON_LOG_INFO("Shutting down due to signal $1: $2", sig, ::sys_siglist[sig]);
    do_exit(exit_success);
  }
  catch(exception& stdex) {
    // Print the message in `stdex`. There isn't much we can do.
    do_exit(exit_system_error, "unhandled exception: %s\n[exception class `%s`]\n",
                               stdex.what(), typeid(stdex).name());
  }
