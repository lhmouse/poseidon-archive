// This file is part of Poseidon.
// Copyleft 2022, LH_Mouse. All wrongs reserved.

#include "precompiled.ipp"
#include "core/config_file.hpp"
#include "static/main_config.hpp"
#include "static/async_logger.hpp"
#include "static/timer_driver.hpp"
#include "static/network_driver.hpp"
#include "static/task_executor_pool.hpp"
#include "static/fiber_scheduler.hpp"
#include "utils.hpp"
#include <locale.h>
#include <signal.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>
#include <dlfcn.h>
#include <pthread.h>
#include <sys/resource.h>
#include <sys/wait.h>
#include <sys/file.h>

namespace {
using namespace poseidon;

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

    ::fflush(nullptr);
    ::quick_exit(0);
  }

[[noreturn]]
int
do_print_version_and_exit()
  {
    ::printf(
//        1         2         3         4         5         6         7     |
// 3456789012345678901234567890123456789012345678901234567890123456789012345|
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""" R"'''''''''''''''(
%s (internal %s)

Visit the homepage at <%s>.
Report bugs to <%s>.
)'''''''''''''''" """"""""""""""""""""""""""""""""""""""""""""""""""""""""+1,
// 3456789012345678901234567890123456789012345678901234567890123456789012345|
//        1         2         3         4         5         6         7     |
      PACKAGE_STRING, POSEIDON_ABI_VERSION_STRING,
      PACKAGE_URL,
      PACKAGE_BUGREPORT);

    ::fflush(nullptr);
    ::quick_exit(0);
  }

// We want to detect Ctrl-C.
atomic_signal exit_sig;

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
::rocket::unique_posix_fd daemon_pipe(::close);
::rocket::unique_posix_fd pid_file(::close);

// These are process exit status codes.
enum Exit_Code : uint8_t
  {
    exit_success            = 0,
    exit_system_error       = 1,
    exit_invalid_argument   = 2,
  };

[[noreturn]] ROCKET_NEVER_INLINE
int
do_exit_printf(Exit_Code code, const char* fmt, ...) noexcept
  {
    // Sleep for a few moments so pending logs are flushed.
    Async_Logger::synchronize();

    // Output the string to standard error.
    ::va_list ap;
    va_start(ap, fmt);
    ::vfprintf(stderr, fmt, ap);
    va_end(ap);

    // Perform fast exit.
    ::fflush(nullptr);
    ::quick_exit(static_cast<int>(code));
  }

ROCKET_NEVER_INLINE
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

ROCKET_NEVER_INLINE
void
do_set_working_directory()
  {
    if(cmdline.cd_here.empty())
      return;

    if(::chdir(cmdline.cd_here.safe_c_str()) != 0)
      POSEIDON_THROW(
          "Could not set working directory to '$2'\n"
          "[`chdir()` failed: $1]",
          format_errno(), cmdline.cd_here);
  }

ROCKET_NEVER_INLINE
void
do_check_euid()
  {
    const auto file = Main_Config::copy();
    const auto qroot = file.get_bool_opt({"general","permit_root_startup"});
    if(qroot && *qroot)
      return;

    if(::geteuid() == 0)
      POSEIDON_THROW(
          "Please do not start this program as root.\n"
          "If you insist, you may set `general.permit_root_startup` in "
          "`main.conf` to `true` to bypass this check. Note that starting "
          "as root is considered insecure. An unprivileged user should have "
          "been created for this service.\n"
          "You have been warned.");
  }

ROCKET_NEVER_INLINE
void
do_daemonize_fork()
  {
    if(!cmdline.daemonize)
      return;

    // If the child process has started up successfully, it should write a
    // message through this pipe. If it is closed without any data received,
    // the parent process shall assume there is an error and wait.
    int pipefds[2];
    if(::pipe(pipefds) != 0)
      POSEIDON_THROW(
          "Could not create pipe for child process\n"
          "[`pipe()` failed: $1]",
          format_errno());

    ::rocket::unique_posix_fd rfd(pipefds[0], ::close);
    daemon_pipe.reset(pipefds[1]);

    // Create the child process now.
    ::fflush(nullptr);
    ::pid_t child = ::fork();
    if(child < 0)
      POSEIDON_THROW(
          "Could not create child process\n"
          "[`fork()` failed: $1]",
          format_errno());

    // If this is the child process, continue execution.
    if(child == 0)
      return;

    // Wait for the notification from the child process. Should one be
    // received, the parent process shall exit. This may be interrupted
    // so we need a loop.
    char dummy[16];
    int wstat;

    for(;;)
      if(::read(rfd, dummy, sizeof(dummy)) >= 0)
        ::_Exit(0);
      else if(errno != EINTR)
        break;

    // Otherwise, wait for the child and forward its exit status.
    // Note `waitpid()` may also return if the child has been stopped
    // or continued.
    for(;;)
      if(::waitpid(child, &wstat, 0) == -1)
        ::_Exit(255);
      else if(WIFEXITED(wstat))
        ::_Exit(WEXITSTATUS(wstat));
      else if(WIFSIGNALED(wstat))
        ::_Exit(128 + WTERMSIG(wstat));
  }

ROCKET_NEVER_INLINE
void
do_init_signal_handlers()
  {
    // Ignore `SIGPIPE` for good.
    struct ::sigaction sigact;
    ::sigemptyset(&(sigact.sa_mask));
    sigact.sa_flags = 0;
    sigact.sa_handler = SIG_IGN;
    ::sigaction(SIGPIPE, &sigact, nullptr);

    // Trap signals. Errors are ignored.
    sigact.sa_handler = [](int n) { exit_sig.store(n);  };
    ::sigaction(SIGINT, &sigact, nullptr);
    ::sigaction(SIGTERM, &sigact, nullptr);
    ::sigaction(SIGHUP, &sigact, nullptr);
    ::sigaction(SIGALRM, &sigact, nullptr);
  }

ROCKET_NEVER_INLINE
void
do_daemonize_finish()
  {
    if(!daemon_pipe)
      return;

    // Notify my parent process. Errors are ignored.
    for(;;)
      if(::write(daemon_pipe, "OK", 2) >= 0)
        break;
      else if(errno != EINTR)
        break;

    // The pipe can be closed now.
    daemon_pipe.reset();
  }

ROCKET_NEVER_INLINE
void
do_write_pid_file()
  {
    const auto file = Main_Config::copy();
    const auto kpath = file.get_string_opt({"general","pid_file_path"});
    if(!kpath || kpath->empty())
      return;

    // Create the lock file.
    pid_file.reset(::creat(kpath->safe_c_str(), 0644));
    if(!pid_file)
      POSEIDON_THROW(
          "Could not create PID file '$2'\n"
          "[`open()` failed: $1]",
          format_errno(), kpath->c_str());

    // Lock it in exclusive mode before overwriting.
    if(::flock(pid_file, LOCK_EX | LOCK_NB) != 0)
      POSEIDON_THROW(
          "Could not lock PID file '$2'\n"
          "(is another instance running?)\n"
          "[`flock()` failed: $1]",
          format_errno(), kpath->c_str());

    // Write the PID of myself.
    POSEIDON_LOG_DEBUG("Writing current process ID to '$1'", kpath->c_str());
    ::dprintf(pid_file, "%d\n", static_cast<int>(::getpid()));

    // Downgrade the lock so the PID may be read by other processes.
    ::flock(pid_file, LOCK_SH);
  }

ROCKET_NEVER_INLINE
void
do_check_ulimits()
  {
    ::rlimit rlim;
    if((::getrlimit(RLIMIT_CORE, &rlim) == 0) && (rlim.rlim_cur <= 0))
      POSEIDON_LOG_WARN(
          "Core dumps are disabled. We highly suggest you enable them in "
          "case of crashes.\n"
          "See `/etc/security/limits.conf` for details.");

    if((::getrlimit(RLIMIT_NOFILE, &rlim) == 0) && (rlim.rlim_cur <= 10'000))
      POSEIDON_LOG_WARN(
          "The limit of number of open files (which is `$1`) is too low. "
          "This might result in denial of service when there are too many "
          "simultaneous network connections. We suggest you set it to least "
          "10,000 for production use.\n"
          "See `/etc/security/limits.conf` for details.",
          rlim.rlim_cur);
  }

ROCKET_NEVER_INLINE
size_t
do_load_addons()
  {
    const auto file = Main_Config::copy();
    const auto qaddons = file.get_array_opt({"general","addons"});
    if(!qaddons || qaddons->empty())
      return 0;

    for(const auto& addon : *qaddons) {
      // Each add-on shall be a path to a shared library to load.
      if(!addon.is_string())
        POSEIDON_LOG_FATAL("Invalid add-on path (`$1` is not a string)", addon);

      const auto& path = addon.as_string();
      POSEIDON_LOG_INFO("Loading add-on: $1", path);

      if(!::dlopen(path.safe_c_str(), RTLD_NOW | RTLD_LOCAL | RTLD_NODELETE))
        POSEIDON_THROW(
            "Failed to load add-on '$1'\n"
            "[`dlopen()` failed: $2]",
            path, ::dlerror());

      POSEIDON_LOG_INFO("Finished loading add-on: $1", path);
    }
    return qaddons->size();
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
    Task_Executor_Pool::reload();
    Fiber_Scheduler::reload();

    POSEIDON_LOG_INFO("Starting up: $1 (PID $2)", PACKAGE_STRING, ::getpid());

    do_check_euid();
    do_daemonize_fork();
    do_init_signal_handlers();
    do_write_pid_file();
    do_check_ulimits();

    if(do_load_addons() == 0)
      POSEIDON_LOG_FATAL("No add-ons have been loaded. What's the job now?");

    // Schedule fibers until a termination signal is caught.
    do_daemonize_finish();
    Fiber_Scheduler::modal_loop(exit_sig);
  }
  catch(exception& stdex) {
    // Print the message in `stdex`. There isn't much we can do.
    do_exit_printf(exit_system_error,
          "%s\n[exception class `%s`]\n", stdex.what(), typeid(stdex).name());
  }
