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
#include "utils.hpp"
#include <locale.h>
#include <signal.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>
#include <dlfcn.h>
#include <sys/resource.h>
#include <sys/wait.h>

namespace {
using namespace poseidon;

[[noreturn]] int
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

    ::fflush(nullptr);
    ::quick_exit(0);
  }

[[noreturn]] int
do_print_version_and_exit()
  {
    ::printf(
//        1         2         3         4         5         6         7     |
// 3456789012345678901234567890123456789012345678901234567890123456789012345|
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""" R"'''''''''''''''(
%s

Visit the homepage at <%s>.
Report bugs to <%s>.
)'''''''''''''''" """"""""""""""""""""""""""""""""""""""""""""""""""""""""+1,
// 3456789012345678901234567890123456789012345678901234567890123456789012345|
//        1         2         3         4         5         6         7     |
      PACKAGE_STRING,
      PACKAGE_URL,
      PACKAGE_BUGREPORT);

    ::fflush(nullptr);
    ::quick_exit(0);
  }

// We want to detect Ctrl-C.
atomic_signal exit_sig;

void
do_trap_exit_signal(int sig) noexcept
  {
    // Trap Ctrl-C. Failure to set the signal handler is ignored.
    struct ::sigaction sigx = { };
    sigx.sa_handler = [](int n) { exit_sig.store(n);  };
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

// They are declared here for convenience.
Command_Line_Options cmdline;
::rocket::unique_posix_fd daemon_wfd(::close);

// These are process exit status codes.
enum Exit_Code : uint8_t
  {
    exit_success            = 0,
    exit_system_error       = 1,
    exit_invalid_argument   = 2,
  };

[[noreturn]] ROCKET_NOINLINE int
do_exit_printf(Exit_Code code, const char* fmt, ...) noexcept
  {
    // Sleep for a few moments so pending logs are flushed.
    Async_Logger::synchronize(1000);

    // Output the string to standard error.
    ::va_list ap;
    va_start(ap, fmt);
    ::vfprintf(stderr, fmt, ap);
    va_end(ap);

    // Perform fast exit.
    ::fflush(nullptr);
    ::quick_exit(static_cast<int>(code));
  }

ROCKET_NOINLINE void
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

        default:
          // `getopt()` will have written a message to standard error.
          do_exit_printf(exit_invalid_argument,
              "Try `%s -h` for help.\n", argv[0]);
      }
    }

    // Check for early exit conditions.
    if(help)
      do_print_help_and_exit(argv[0]);

    if(version)
      do_print_version_and_exit();

    // If more arguments follow, they denote the working directory.
    if(argc - optind > 1)
      do_exit_printf(exit_invalid_argument,
          "%s: too many arguments -- '%s'\n"
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

ROCKET_NOINLINE void
do_set_working_directory()
  {
    if(cmdline.cd_here.empty())
      return;

    if(::chdir(cmdline.cd_here.safe_c_str()) != 0)
      POSEIDON_THROW("Could not set working directory to '$2'\n"
                     "[`chdir()` failed: $1]",
                     format_errno(errno), cmdline.cd_here);
  }

ROCKET_NOINLINE void
do_check_euid()
  {
    const auto file = Main_Config::copy();
    const auto qroot = file.get_bool_opt({"general","permit_root_startup"});
    if(qroot && *qroot)
      return;

    if(::geteuid() == 0)
      POSEIDON_LOG_ERROR(
          "Please do not start this program as root.\n"
          "If you insist, you may set `general.permit_root_startup` in `main.conf` "
          "to `true` to bypass this check. Note that starting as root is considered "
          "insecure. An unprivileged user should have been created for this service.\n"
          "You have been warned.");
  }

ROCKET_NOINLINE void
do_daemonize_fork()
  {
    if(!cmdline.daemonize)
      return;

    // If the child process has started up successfully, it should write a message
    // through this pipe. If it is closed without any data received, the parent
    // process shall assume there is an error and wait.
    int pipefds[2];
    if(::pipe(pipefds) != 0)
      POSEIDON_THROW("Could not create pipe for child process\n"
                     "[`pipe()` failed: $1]",
                     format_errno(errno));

    const ::rocket::unique_posix_fd rfd(pipefds[0], ::close);
    daemon_wfd.reset(pipefds[1]);

    ::pid_t child = ::fork();
    if(child < 0)
      POSEIDON_THROW("Could not create child process\n"
                     "[`fork()` failed: $1]",
                     format_errno(errno));

    // Return in the child process.
    if(child == 0)
      return;

    // This is the parent process.
    daemon_wfd.reset();
    ::fflush(nullptr);

    // Wait for the notification from the child process.
    ::ssize_t nread;
    do {
      char msg[64];
      nread = ::read(rfd, msg, sizeof(msg));
    }
    while((nread < 0) && (errno == EINTR));

    // If a response has been received, exit normally.
    if(nread > 0)
      ::_Exit(0);

    // Otherwise, wait for the child and forward its exit status.
    // Note: `waitpid()` may return if the child has been stopped or continued.
    do {
      int wstat;
      if(::waitpid(child, &wstat, 0) == -1)
        ::_Exit(255);

      if(WIFEXITED(wstat))
        ::_Exit(WEXITSTATUS(wstat));

      if(WIFSIGNALED(wstat))
        ::_Exit(128 + WTERMSIG(wstat));
    }
    while(true);
  }

ROCKET_NOINLINE void
do_daemonize_finish()
  {
    if(!daemon_wfd)
      return;

    // Notify my parent process. Errors are ignored.
    ::ssize_t nwritten;
    do {
      static const char msg[] = "OK";
      nwritten = ::write(daemon_wfd, msg, sizeof(msg));
    }
    while((nwritten < 0) && (errno == EINTR));

    if(nwritten < 0)
      POSEIDON_LOG_DEBUG("Failed to notify parent process: $1",
                         format_errno(errno));

    daemon_wfd.reset();
  }

ROCKET_NOINLINE void
do_check_ulimits()
  {
    ::rlimit rlim;
    if((::getrlimit(RLIMIT_CORE, &rlim) == 0) && (rlim.rlim_cur <= 0))
      POSEIDON_LOG_WARN(
          "Core dumps are disabled. We suggest you enable them in case of crashes.\n"
          "See `/etc/security/limits.conf` for details.");

    if((::getrlimit(RLIMIT_NOFILE, &rlim) == 0) && (rlim.rlim_cur <= 10'000))
      POSEIDON_LOG_WARN(
          "The limit of number of open files (which is `$1`) is too low. This might "
          "result in denial of service when there are too many simultaneous network "
          "connections. We suggest you set it to least 10,000 for production use.\n"
          "See `/etc/security/limits.conf` for details.",
          rlim.rlim_cur);
  }

ROCKET_NOINLINE void
do_load_addons()
  {
    const auto file = Main_Config::copy();
    const auto qaddons = file.get_array_opt({"general","addons"});
    if(!qaddons || qaddons->empty()) {
      POSEIDON_LOG_FATAL("There is no add-on to load. What's the job now?");
      return;
    }

    for(const auto& addon : *qaddons) {
      // Each add-on shall be a path to a shared library to load.
      if(!addon.is_string())
        POSEIDON_LOG_FATAL("Invalid add-on path (`$1` is not a string)", addon);

      const auto& path = addon.as_string();
      POSEIDON_LOG_INFO("Loading add-on: $1", path);

      if(!::dlopen(path.safe_c_str(), RTLD_NOW | RTLD_LOCAL | RTLD_NODELETE))
        POSEIDON_THROW("Error loading add-on '$1'\n"
                       "[`dlopen()` failed: $2]",
                       path, ::dlerror());

      POSEIDON_LOG_INFO("Finished loading add-on: $1", path);
    }
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
    do_set_working_directory();

    // Load 'main.conf' before daemonization, so any earlier failures are
    // visible to the user.
    Main_Config::reload();
    Async_Logger::reload();
    Network_Driver::reload();
    Worker_Pool::reload();
    Fiber_Scheduler::reload();

    do_check_euid();
    do_daemonize_fork();

    // Set name of the main thread. Failure to set the name is ignored.
    ::pthread_setname_np(::pthread_self(), "poseidon");

    // Disable cancellation for safety. Failure to set the cancel state is ignored.
    ::pthread_setcancelstate(PTHREAD_CANCEL_DISABLE, nullptr);

    // Trap exit signals. Failure to set signal handlers is ignored.
    // This also makes stdio functions fail immediately.
    do_trap_exit_signal(SIGINT);
    do_trap_exit_signal(SIGTERM);
    do_trap_exit_signal(SIGHUP);
    do_trap_exit_signal(SIGALRM);

    // Ignore `SIGPIPE` for good.
    ::signal(SIGPIPE, SIG_IGN);

    // Start daemon threads.
    POSEIDON_LOG_INFO("Starting up: $1 (PID $2)", PACKAGE_STRING, ::getpid());

    Async_Logger::start();
    Timer_Driver::start();
    Network_Driver::start();

    do_check_ulimits();
    do_load_addons();

    POSEIDON_LOG_INFO("Startup complete: $1 (PID $2)", PACKAGE_STRING, ::getpid());

    // Schedule fibers until a termination signal is caught.
    do_daemonize_finish();
    Fiber_Scheduler::modal_loop(exit_sig);
  }
  catch(exception& stdex) {
    // Print the message in `stdex`. There isn't much we can do.
    do_exit_printf(exit_system_error,
        "%s\n[exception class `%s`]\n", stdex.what(), typeid(stdex).name());
  }
