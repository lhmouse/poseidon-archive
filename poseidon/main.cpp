// This file is part of Poseidon.
// Copyleft 2022, LH_Mouse. All wrongs reserved.

#include "precompiled.ipp"
#include "base/config_file.hpp"
#include "static/main_config.hpp"
#include "static/fiber_scheduler.hpp"
#include "static/async_logger.hpp"
#include "static/timer_driver.hpp"
#include "static/async_task_executor.hpp"
#include "static/network_driver.hpp"
#include "utils.hpp"
#include <locale.h>
#include <signal.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>
#include <dlfcn.h>
#include <pthread.h>
#include <sys/file.h>
#include <sys/resource.h>

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

  -h      show help message then exit
  -V      show version information then exit
  -v      enable verbose mode

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

// Define command-line options here.
struct Command_Line_Options
  {
    // options
    bool verbose = false;

    // non-options
    cow_string cd_here;
  };

// They are declared here for convenience.
Command_Line_Options cmdline;

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
    // Wait for pending logs to be flushed.
    async_logger.synchronize();

    // Output the string to standard error.
    ::va_list ap;
    va_start(ap, fmt);
    ::vfprintf(stderr, fmt, ap);
    va_end(ap);

    // Perform fast exit.
    ::fflush(nullptr);
    ::quick_exit((int)code);
  }

ROCKET_NEVER_INLINE
void
do_parse_command_line(int argc, char** argv)
  {
    bool help = false;
    bool version = false;

    optional<bool> verbose;
    optional<cow_string> cd_here;

    // Check for some common options before calling `getopt()`.
    if(argc > 1) {
      if(::strcmp(argv[1], "--help") == 0)
        do_print_help_and_exit(argv[0]);

      if(::strcmp(argv[1], "--version") == 0)
        do_print_version_and_exit();
    }

    // Parse command-line options.
    int ch;
    while((ch = ::getopt(argc, argv, "+hVv")) != -1) {
      switch(ch) {
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
      POSEIDON_THROW((
          "Could not set working directory to '$2'",
          "[`chdir()` failed: $1]"),
          format_errno(), cmdline.cd_here);
  }

template<class ObjectT>
void
do_create_resident_thread(ObjectT& obj, const char* name)
  {
    auto thrd_function = +[](void* ptr) noexcept
      {
        // Set thread information. Errors are ignored.
        int oldst;
        ::pthread_setcancelstate(PTHREAD_CANCEL_DISABLE, &oldst);

        ::sigset_t sigset;
        ::sigemptyset(&sigset);
        ::sigaddset(&sigset, SIGINT);
        ::sigaddset(&sigset, SIGTERM);
        ::sigaddset(&sigset, SIGHUP);
        ::sigaddset(&sigset, SIGALRM);
        ::pthread_sigmask(SIG_BLOCK, &sigset, nullptr);

        // Enter an infinite loop.
        for(;;)
          try {
            ((ObjectT*) ptr)->thread_loop();
          }
          catch(exception& stdex) {
            ::fprintf(stderr,
                "WARNING: Caught an exception from thread loop: %s\n"
                "[static class `%s`]\n"
                "[exception class `%s`]\n",
                stdex.what(), typeid(ObjectT).name(), typeid(stdex).name());
          }

        // Make the return type deducible.
        return (void*) nullptr;
      };

    ::pthread_t thrd;
    int err = ::pthread_create(&thrd, nullptr, thrd_function, ::std::addressof(obj));
    if(err != 0)
      ::rocket::sprintf_and_throw<::std::runtime_error>(
          "Could not create thread: %s\n"
          "[`pthread_create()` failed: %d]"
          "[static class `%s`]\n",
          ::strerror(err), err, typeid(ObjectT).name());

    // Name the thread and detach it. Errors are ignored.
    ::pthread_setname_np(thrd, name);
    ::pthread_detach(thrd);
  }

ROCKET_NEVER_INLINE
void
do_create_threads()
  {
    do_create_resident_thread(async_logger, "logger");
    do_create_resident_thread(timer_driver, "timer");
    do_create_resident_thread(async_task_executor, "task1");
    do_create_resident_thread(async_task_executor, "task2");
    do_create_resident_thread(async_task_executor, "task3");
    do_create_resident_thread(async_task_executor, "task4");
    do_create_resident_thread(network_driver, "network");
  }

ROCKET_NEVER_INLINE
void
do_check_euid()
  {
    bool permit_root_startup = false;
    const auto conf = main_config.copy();

    auto value = conf.query("general", "permit_root_startup");
    if(value.is_boolean())
      permit_root_startup = value.as_boolean();
    else if(!value.is_null())
      POSEIDON_LOG_WARN((
          "Ignoring `general.permit_root_startup`: expecting a `boolean`, got `$1`",
          "[in configuration file '$2']"),
          value, conf.path());

    if(!permit_root_startup && (::geteuid() == 0))
      POSEIDON_THROW((
          "Please do not start this program as root. If you insist, you may "
          "set `general.permit_root_startup` in `$1` to `true` to bypass this "
          "check. Note that starting as root should be considered insecure. An "
          "unprivileged user should have been created for this service. You "
          "have been warned."),
          conf.path());
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
    sigact.sa_handler = +[](int n) { exit_signal.store(n);  };
    ::sigaction(SIGINT, &sigact, nullptr);
    ::sigaction(SIGTERM, &sigact, nullptr);
    ::sigaction(SIGHUP, &sigact, nullptr);
    ::sigaction(SIGALRM, &sigact, nullptr);
  }

ROCKET_NEVER_INLINE
void
do_write_pid_file()
  {
    cow_string pid_file_path;
    const auto conf = main_config.copy();

    auto value = conf.query("general", "pid_file_path");
    if(value.is_string())
      pid_file_path = value.as_string();
    else if(!value.is_null())
      POSEIDON_LOG_WARN((
          "Ignoring `general.permit_root_startup`: expecting a `string`, got `$1`",
          "[in configuration file '$2']"),
          value, conf.path());

    if(pid_file_path.empty())
      return;

    // Create the lock file and lock it in exclusive mode before overwriting.
    unique_posix_fd pid_file(::creat(pid_file_path.safe_c_str(), 0644));
    if(!pid_file)
      POSEIDON_THROW((
          "Could not create PID file '$2'",
          "[`open()` failed: $1]"),
          format_errno(), pid_file_path.c_str());

    if(::flock(pid_file, LOCK_EX | LOCK_NB) != 0)
      POSEIDON_THROW((
          "Could not lock PID file '$2' because it is being locked by another process",
          "[`flock()` failed: $1]"),
          format_errno(), pid_file_path.c_str());

    // Write the PID of myself.
    POSEIDON_LOG_DEBUG(("Writing current process ID to '$1'"), pid_file_path.c_str());
    ::dprintf(pid_file, "%d\n", (int) ::getpid());

    // Downgrade the lock so the PID may be read by others.
    ::flock(pid_file, LOCK_SH);
  }

ROCKET_NEVER_INLINE
void
do_check_ulimits()
  {
    ::rlimit rlim;
    if((::getrlimit(RLIMIT_CORE, &rlim) == 0) && (rlim.rlim_cur <= 0))
      POSEIDON_LOG_WARN((
          "Core dumps have been disabled. We highly suggest you enable them in case "
          "of crashes. See `/etc/security/limits.conf` for details."));

    if((::getrlimit(RLIMIT_NOFILE, &rlim) == 0) && (rlim.rlim_cur <= 10'000))
      POSEIDON_LOG_WARN((
          "The limit of number of open files (which is `$1`) is too low. This might "
          "result in denial of service when there are too many simultaneous network "
          "connections. We suggest you set it to least `10000` for production use. "
          "See `/etc/security/limits.conf` for details."),
          rlim.rlim_cur);
  }

ROCKET_NEVER_INLINE
void
do_load_addons()
  {
    cow_vector<::asteria::Value> addons;
    size_t count = 0;
    const auto conf = main_config.copy();

    auto value = conf.query("addons");
    if(value.is_array())
      addons = value.as_array();
    else if(!value.is_null())
      POSEIDON_LOG_WARN((
          "Ignoring `addons`: expecting an `array`, got `$1`",
          "[in configuration file '$2']"),
          value, conf.path());

    for(const auto& addon : addons) {
      cow_string path;

      if(addon.is_string())
        path = addon.as_string();
      else if(!addon.is_null())
        POSEIDON_LOG_WARN((
            "Ignoring invalid path to add-on: $1",
            "[in configuration file '$2']"),
            addon, conf.path());

      if(path.empty())
        continue;

      POSEIDON_LOG_INFO(("Loading add-on: $1"), path);

      if(::dlopen(path.safe_c_str(), RTLD_NOW | RTLD_NODELETE))
        count ++;
      else
        POSEIDON_LOG_ERROR((
            "Failed to load add-on: $1",
            "[`dlopen()` failed: $2]"),
            path, ::dlerror());

      POSEIDON_LOG_INFO(("Finished loading add-on: $1"), path);
    }

    if(count == 0)
      POSEIDON_LOG_FATAL(("No add-on has been loaded. What's the job now?"));
  }

}  // namespace

int
main(int argc, char** argv)
  try {
    // Select the C locale.
    // UTF-8 is required for wide-oriented standard streams.
    ::setlocale(LC_ALL, "C.UTF-8");
    ::tzset();
    ::pthread_setname_np(::pthread_self(), PACKAGE);
    ::pthread_setcancelstate(PTHREAD_CANCEL_DISABLE, &::opterr);
    ::srandom((uint32_t) ::clock());

    // Note that this function shall not return in case of errors.
    do_parse_command_line(argc, argv);

    // Load configuration and start the logger early.
    do_set_working_directory();
    main_config.reload();
    do_create_threads();

    async_logger.reload(main_config.copy());
    fiber_scheduler.reload(main_config.copy());
    network_driver.reload(main_config.copy());
    POSEIDON_LOG_INFO(("Finished loading configuration"));

    do_check_euid();
    do_check_ulimits();
    do_init_signal_handlers();
    do_write_pid_file();
    do_load_addons();
    POSEIDON_LOG_INFO(("Startup complete: $1"), PACKAGE_STRING);

    // Schedule fibers until a signal has been received and the scheduler is empty.
    while(fiber_scheduler.size() || !exit_signal.load())
      fiber_scheduler.thread_loop();

    int sig = exit_signal.load();
    POSEIDON_LOG_INFO(("Shutting down due to signal $1: $2"), sig, ::strsignal(sig));
    do_exit_printf(exit_success, "");
  }
  catch(exception& stdex) {
    // Print the message in `stdex`. There isn't much we can do.
    do_exit_printf(exit_system_error,
        "%s\n[exception class `%s`]\n",
        stdex.what(), typeid(stdex).name());
  }
