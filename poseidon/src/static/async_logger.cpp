// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "async_logger.hpp"
#include "main_config.hpp"
#include "../core/config_file.hpp"
#include "../xutilities.hpp"
#include <sys/syscall.h>

namespace poseidon {
namespace {

struct Level_Config
  {
    cow_string color;
    int out_fd = -1;
    cow_string out_path;
  };

void
do_load_level_config(Level_Config& conf, const Config_File& file, const char* name)
  {
    // Read color settings.
    // If we decide to disable color later on, we clear this string.
    if(const auto qcolor = file.get_string_opt({"logger","levels",name,"color"}))
      conf.color = ::std::move(*qcolor);

    // Read stream settings.
    if(const auto qstrm = file.get_string_opt({"logger","levels",name,"stream"})) {
      const auto strm = qstrm->safe_c_str();

      // Set standard file descriptors.
      if(*qstrm == "stdout")
        conf.out_fd = STDOUT_FILENO;

      if(*qstrm == "stderr")
        conf.out_fd = STDERR_FILENO;

      // Read the color setting for this stream.
      bool real_color;
      if(const auto qcolor = file.get_bool_opt({"logger","streams",strm,"color"}))
        real_color = *qcolor;
      else
        real_color = ::isatty(conf.out_fd);

      if(!real_color)
        conf.color.clear();

      // Read the alternative output file.
      if(const auto qfile = file.get_string_opt({"logger","streams",strm,"file"}))
        conf.out_path = ::std::move(*qfile);
      else
        conf.out_path = ::rocket::sref("");
    }
  }

struct Entry
  {
    Log_Level level;
    const char* file;
    long line;
    const char* func;
    cow_string text;

    char thr_name[16];  // thread name
    ::pid_t thr_lwpid;  // kernel thread (LWP) ID, not pthread_t
  };

constexpr char s_escapes[][8] =
  {
    "\\0",   "\\x01", "\\x02", "\\x03", "\\x04", "\\x05", "\\x06", "\\a",
    "\\b",   "\t",    "\n\t",  "\\v",   "\\f",   "\\r",   "\\x0E", "\\x0F",
    "\\x10", "\\x11", "\\x12", "\\x13", "\\x14", "\\x15", "\\x16", "\\x17",
    "\\x18", "\\x19", "\\x1A", "\\x1B", "\\x1C", "\\x1D", "\\x1E", "\\x1F",
    " ",     "!",     "\"",    "#",     "$",     "%",     "&",     "\'",
    "(",     ")",     "*",     "+",     ",",     "-",     ".",     "/",
    "0",     "1",     "2",     "3",     "4",     "5",     "6",     "7",
    "8",     "9",     ":",     ";",     "<",     "=",     ">",     "?",
    "@",     "A",     "B",     "C",     "D",     "E",     "F",     "G",
    "H",     "I",     "J",     "K",     "L",     "M",     "N",     "O",
    "P",     "Q",     "R",     "S",     "T",     "U",     "V",     "W",
    "X",     "Y",     "Z",     "[",     "\\",    "]",     "^",     "_",
    "`",     "a",     "b",     "c",     "d",     "e",     "f",     "g",
    "h",     "i",     "j",     "k",     "l",     "m",     "n",     "o",
    "p",     "q",     "r",     "s",     "t",     "u",     "v",     "w",
    "x",     "y",     "z",     "{",     "|",     "}",     "~",     "\\x7F",
    "\x80",  "\x81",  "\x82",  "\x83",  "\x84",  "\x85",  "\x86",  "\x87",
    "\x88",  "\x89",  "\x8A",  "\x8B",  "\x8C",  "\x8D",  "\x8E",  "\x8F",
    "\x90",  "\x91",  "\x92",  "\x93",  "\x94",  "\x95",  "\x96",  "\x97",
    "\x98",  "\x99",  "\x9A",  "\x9B",  "\x9C",  "\x9D",  "\x9E",  "\x9F",
    "\xA0",  "\xA1",  "\xA2",  "\xA3",  "\xA4",  "\xA5",  "\xA6",  "\xA7",
    "\xA8",  "\xA9",  "\xAA",  "\xAB",  "\xAC",  "\xAD",  "\xAE",  "\xAF",
    "\xB0",  "\xB1",  "\xB2",  "\xB3",  "\xB4",  "\xB5",  "\xB6",  "\xB7",
    "\xB8",  "\xB9",  "\xBA",  "\xBB",  "\xBC",  "\xBD",  "\xBE",  "\xBF",
    "\\xC0", "\\xC1", "\xC2",  "\xC3",  "\xC4",  "\xC5",  "\xC6",  "\xC7",
    "\xC8",  "\xC9",  "\xCA",  "\xCB",  "\xCC",  "\xCD",  "\xCE",  "\xCF",
    "\xD0",  "\xD1",  "\xD2",  "\xD3",  "\xD4",  "\xD5",  "\xD6",  "\xD7",
    "\xD8",  "\xD9",  "\xDA",  "\xDB",  "\xDC",  "\xDD",  "\xDE",  "\xDF",
    "\xE0",  "\xE1",  "\xE2",  "\xE3",  "\xE4",  "\xE5",  "\xE6",  "\xE7",
    "\xE8",  "\xE9",  "\xEA",  "\xEB",  "\xEC",  "\xED",  "\xEE",  "\xEF",
    "\xF0",  "\xF1",  "\xF2",  "\xF3",  "\xF4",  "\\xF5", "\\xF6", "\\xF7",
    "\\xF8", "\\xF9", "\\xFA", "\\xFB", "\\xFC", "\\xFD", "\\xFE", "\\xFF",
  };

template<typename... ParamsT>
bool
do_color(tinyfmt& fmt, const Level_Config& conf, const ParamsT&... params)
  {
    if(conf.color.empty())
      return false;

    void* unused[] = { &(fmt << "\x1B["), &(fmt << params)..., &(fmt << 'm') };
    (void)unused;
    return true;
  }

bool
do_end_color(tinyfmt& fmt, const Level_Config& conf)
  {
    if(conf.color.empty())
      return false;

    fmt << "\x1B[0m";
    return true;
  }

struct Level_Name
  {
    char conf_name[8];
    char fmt_name[8];
  }
constexpr s_level_names[] =
  {
    { "fatal",  " FATAL "  },
    { "error",  " ERROR "  },
    { "warn",   " WARN  "  },
    { "info",   " INFO  "  },
    { "debug",  " DEBUG "  },
    { "trace",  " TRACE "  },
  };

using Level_Config_Array = array<Level_Config, ::rocket::countof(s_level_names)>;

const char*
do_write_loop(int fd, const char* data, size_t size)
  {
    auto bp = static_cast<const char*>(data);
    const auto ep = bp + size;

    for(;;) {
      ::size_t nrem = size_t(ep - bp);
      if(nrem == 0)
        break;  // succeed

      ::ssize_t nwrtn;
      do
        nwrtn = ::write(fd, bp, nrem);
      while((nwrtn < 0) && (errno == EINTR));

      if(nwrtn <= 0)
        break;  // fail

      bp += nwrtn;
    }
    return bp;
  }

void
do_write_log_entry(const Level_Config& conf, Entry&& entry)
  {
    // Get list of streams to write.
    ::rocket::static_vector<::rocket::unique_posix_fd, 4> strms;
    const auto& names = s_level_names[entry.level];

    if(conf.out_fd != -1)
      strms.emplace_back(conf.out_fd, nullptr);  // don't close

    if(conf.out_path.size()) {
      ::rocket::unique_posix_fd fd(::open(conf.out_path.c_str(),
                                          O_WRONLY | O_APPEND | O_CREAT, 0666),
                                   ::close);
      if(fd != -1)
        strms.emplace_back(::std::move(fd));
      else
        ::std::fprintf(stderr,
            "WARNING:could not open log file '%s': error %d: %m\n",
            conf.out_path.c_str(), errno);
    }

    // If no stream is opened, return immediately.
    if(strms.empty())
      return;

    // Compose the string to write.
    ::rocket::tinyfmt_str fmt;
    fmt.set_string(cow_string(2047, '/'));
    fmt.clear_string();

    // Write the timestamp.
    ::timespec ts;
    ::clock_gettime(CLOCK_REALTIME, &ts);
    ::tm tr;
    ::localtime_r(&(ts.tv_sec), &tr);

    do_color(fmt, conf, conf.color);
    ::rocket::ascii_numput nump;
    // 'yyyy-mmmm-dd HH:MM:SS.sss'
    fmt << nump.put_DU(static_cast<uint64_t>(tr.tm_year + 1900), 4);
    fmt << '-' << nump.put_DU(static_cast<uint64_t>(tr.tm_mon + 1), 2);
    fmt << '-' << nump.put_DU(static_cast<uint64_t>(tr.tm_mday), 2);
    fmt << ' ' << nump.put_DU(static_cast<uint64_t>(tr.tm_hour), 2);
    fmt << ':' << nump.put_DU(static_cast<uint64_t>(tr.tm_min), 2);
    fmt << ':' << nump.put_DU(static_cast<uint64_t>(tr.tm_sec), 2);
    fmt << '.' << nump.put_DU(static_cast<uint64_t>(ts.tv_nsec), 9);
    do_end_color(fmt, conf);
    fmt << ' ';

    // Write the log level string.
    do_color(fmt, conf, conf.color, ";7");  // reverse
    fmt << names.fmt_name;
    do_end_color(fmt, conf);
    fmt << ' ';

    // Write the thread ID and name.
    do_color(fmt, conf, "30;1");  // grey
    fmt << "Thread \"" << entry.thr_name << "\" [LWP " << entry.thr_lwpid << "]";
    do_end_color(fmt, conf);
    fmt << ' ';

    // Write the function name.
    do_color(fmt, conf, "37;1");  // bright white
    fmt << "Function `" << entry.func << "`";
    do_end_color(fmt, conf);
    fmt << ' ';

    // Write the file name and line number.
    do_color(fmt, conf, "34;1");  // bright blue
    fmt << "@ " << entry.file << ':' << entry.line;
    do_end_color(fmt, conf);
    fmt << "\n\t";

    // Write the message.
    do_color(fmt, conf, conf.color);
    for(size_t k = 0;  k != entry.text.size();  ++k) {
      const auto& seq = s_escapes[uint8_t(entry.text[k])];
      if(ROCKET_EXPECT(seq[1] == 0)) {
        // Optimize the operation a little if it is a non-escaped character.
        fmt << seq[0];
      }
      else if(ROCKET_EXPECT(seq[0] != '\\')) {
        // Insert non-escaped characters verbatim.
        fmt << seq;
      }
      else {
        // Write an escaped sequence.
        do_color(fmt, conf, "7");  // reverse
        fmt << seq;
        do_color(fmt, conf, "27");  // reverse
      }
    }
    do_end_color(fmt, conf);
    fmt << '\n';

    auto str = fmt.extract_string();

    // Write data to all streams.
    for(const auto& fd : strms)
      do_write_loop(fd, str.data(), str.size());
  }

}  // namespace

POSEIDON_STATIC_CLASS_DEFINE(Async_Logger)
  {
    // constant data
    ::rocket::once_flag m_init_once;
    ::pthread_t m_thread = 0;

    // configuration
    mutable mutex m_conf_mutex;
    Level_Config_Array m_conf_levels;

    // dynamic data
    mutable mutex m_queue_mutex;
    condition_variable m_queue_avail;
    ::std::deque<Entry> m_queue;

    static
    void
    do_init_once()
      {
        // Create the thread. Note it is never joined or detached.
        mutex::unique_lock lock(self->m_queue_mutex);
        self->m_thread = noadl::create_daemon_thread<do_thread_loop>("logger");
      }

    static
    void
    do_thread_loop(void* /*param*/)
      {
        Entry entry;
        bool needs_sync;

        // Await an entry and pop it.
        mutex::unique_lock lock(self->m_queue_mutex);
        for(;;) {
          if(self->m_queue.empty()) {
            // Wait until an entry becomes available.
            self->m_queue_avail.wait(lock);
            continue;
          }

          // Pop it.
          entry = ::std::move(self->m_queue.front());
          self->m_queue.pop_front();
          needs_sync = (entry.level < log_level_warn) || self->m_queue.empty();
          break;
        }
        lock.unlock();

        // Get configuration for this level.
        lock.assign(self->m_conf_mutex);
        const auto conf = self->m_conf_levels.at(entry.level);
        lock.unlock();

        // Write this entry.
        do_write_log_entry(conf, ::std::move(entry));

        if(needs_sync)
          ::sync();
      }
  };

void
Async_Logger::
start()
  {
    self->m_init_once.call(self->do_init_once);
  }

void
Async_Logger::
reload()
  {
    // Load logger settings into this temporary array.
    auto file = Main_Config::copy();

    Level_Config_Array temp;
    for(size_t k = 0;  k < temp.size();  ++k)
      do_load_level_config(temp[k], file, s_level_names[k].conf_name);

    // During destruction of `temp` the mutex should have been unlocked.
    // The swap operation is presumed to be fast, so we don't hold the mutex
    // for too long.
    mutex::unique_lock lock(self->m_conf_mutex);
    self->m_conf_levels.swap(temp);
  }

bool
Async_Logger::
is_enabled(Log_Level level)
noexcept
  {
    // Validate arguments.
    mutex::unique_lock lock(self->m_conf_mutex);
    if(level >= self->m_conf_levels.size())
      return false;

    // The level is enabled if at least one stream is enabled.
    const auto& conf = self->m_conf_levels[level];
    return (conf.out_fd != -1) || conf.out_path.size();
  }

size_t
Async_Logger::
queue_size()
noexcept
  {
    // Return the number of pending entries.
    mutex::unique_lock lock(self->m_conf_mutex);
    return self->m_queue.size();
  }

bool
Async_Logger::
enqueue(Log_Level level, const char* file, long line, const char* func,
        cow_string&& text)
  {
    if(level >= self->m_conf_levels.size())
      return false;

    // Compose the entry.
    Entry entry = { level, file, line, func, ::std::move(text), "", 0 };
    ::pthread_getname_np(::pthread_self(), entry.thr_name, sizeof(entry.thr_name));
    entry.thr_lwpid = static_cast<::pid_t>(::syscall(__NR_gettid));

    if(ROCKET_UNEXPECT(self->m_thread == 0)) {
      // If the logger thread has not been created, write it immediately.
      const auto& conf = self->m_conf_levels.at(entry.level);
      do_write_log_entry(conf, ::std::move(entry));
      return true;
    }

    // Push the entry.
    mutex::unique_lock lock(self->m_queue_mutex);
    self->m_queue.emplace_back(::std::move(entry));
    self->m_queue_avail.notify_one();
    return true;
  }

}  // namespace poseidon
