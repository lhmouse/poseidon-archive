// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "async_logger.hpp"
#include "main_config.hpp"
#include "../core/config_file.hpp"
#include "../utils.hpp"
#include <sys/syscall.h>

namespace poseidon {
namespace {

struct Level_Name
  {
    char conf_name[8];
    char fmt_name[8];
  }
constexpr s_level_names[] =
  {
    { "fatal",  "[FATAL]"  },
    { "error",  "[ERROR]"  },
    { "warn",   "[WARN ]"  },
    { "info",   "[INFO ]"  },
    { "debug",  "[DEBUG]"  },
    { "trace",  "[TRACE]"  },
  };

struct Level_Config
  {
    cow_string color;
    int out_fd = -1;
    cow_string out_path;
    bool trivial = false;
  };

constexpr size_t level_count = ::rocket::size(s_level_names);
using Level_Config_Array = array<Level_Config, level_count>;

void
do_load_level_config(Level_Config& conf, const Config_File& file, const char* name)
  {
    // Read color settings.
    // If we decide to disable color later on, we clear this string.
    auto qstr = file.get_string_opt({"logger","levels",name,"color"});
    if(qstr)
      conf.color = ::std::move(*qstr);

    // Read stream settings.
    qstr = file.get_string_opt({"logger","levels",name,"stream"});
    if(qstr) {
      const auto strm_str = ::std::move(*qstr);
      auto strm = strm_str.safe_c_str();

      // Set standard file descriptors.
      if(::strcmp(strm, "stdout") == 0)
        conf.out_fd = STDOUT_FILENO;

      if(::strcmp(strm, "stderr") == 0)
        conf.out_fd = STDERR_FILENO;

      // Read the color setting for this stream.
      // If no explicit `true` or `false` is given, color is enabled if
      // the file descriptor denotes a terminal.
      auto qcolor = file.get_bool_opt({"logger","streams",strm,"color"});
      if(!qcolor)
        qcolor = (conf.out_fd >= 0) && ::isatty(conf.out_fd);

      if(!*qcolor)
        conf.color.clear();

      // Read the alternative output file.
      qstr = file.get_string_opt({"logger","streams",strm,"file"});
      if(!qstr)
        qstr.emplace();

      conf.out_path = ::std::move(*qstr);
    }

    // Is this level trivial?
    auto qbool = file.get_bool_opt({"logger","levels",name,"trivial"});
    if(qbool)
      conf.trivial = *qbool;
  }

struct Log_Entry
  {
    Log_Level level;
    const char* file;
    long line;
    const char* func;
    cow_string text;

    ::pid_t thr_lwpid;  // kernel thread (LWP) ID, not pthread_t
    char thr_name[16];  // thread name
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
do_color(cow_string& log_text, const Level_Config& conf, const ParamsT&... params)
  {
    if(conf.color.empty())
      return false;

    log_text += "\x1B[";
    (void*[]){ &(log_text += params)... };
    log_text += "m";
    return true;
  }

bool
do_end_color(cow_string& log_text, const Level_Config& conf)
  {
    if(conf.color.empty())
      return false;

    log_text += "\x1B[0m";
    return true;
  }

bool
do_write_log_entry(const Level_Config_Array& conf_levels, Log_Entry&& entry)
  {
    // Get list of streams to write.
    const auto& conf = conf_levels.at(entry.level);
    ::rocket::static_vector<::rocket::unique_posix_fd, 4> strms;
    const auto& names = s_level_names[entry.level];

    if(conf.out_fd != -1)
      strms.emplace_back(conf.out_fd, nullptr);  // don't close

    if(conf.out_path.size()) {
      ::rocket::unique_posix_fd fd(::open(conf.out_path.c_str(),
                         O_WRONLY | O_APPEND | O_CREAT, 0666), ::close);
      if(fd != -1)
        strms.emplace_back(::std::move(fd));
      else
        ::std::fprintf(stderr,
            "WARNING: Could not open log file '%s': error %d: %m\n",
            conf.out_path.c_str(), errno);
    }

    // If no stream is opened, return immediately.
    if(strms.empty())
      return false;

    // Compose the string to write.
    cow_string log_text;
    log_text.reserve(2047);

    // Write the timestamp and log level string.
    ::timespec ts;
    ::clock_gettime(CLOCK_REALTIME, &ts);
    ::tm tr;
    ::localtime_r(&(ts.tv_sec), &tr);

    do_color(log_text, conf, conf.color);
    char temp[64];
    ::strftime(temp, sizeof(temp), "%F %R", &tr);
    log_text += temp;
    ::sprintf(temp, ":%2.9f ", tr.tm_sec + double(ts.tv_nsec) / 1.0e9);
    log_text += temp;
    do_color(log_text, conf, "7");  // reverse
    log_text += names.fmt_name;
    do_end_color(log_text, conf);
    log_text += "  ";

    // Write the thread ID and name.
    do_color(log_text, conf, "30;1");  // grey
    log_text += "Thread \"";
    log_text += entry.thr_name;
    ::sprintf(temp, "\" [LWP %ld]", long(entry.thr_lwpid));
    log_text += temp;
    do_end_color(log_text, conf);
    log_text += "  ";

    // Write the function name.
    do_color(log_text, conf, "37;1");  // bright white
    log_text += "Function `";
    log_text += entry.func;
    log_text += "`";
    do_end_color(log_text, conf);
    log_text += "  ";

    // Write the file name and line number.
    do_color(log_text, conf, "34;1");  // bright blue
    log_text += "@ ";
    log_text += entry.file;
    ::sprintf(temp, ":%ld", entry.line);
    log_text += temp;
    do_end_color(log_text, conf);
    log_text += "\n\t";

    // Write the message.
    do_color(log_text, conf, conf.color);
    for(size_t k = 0;  k != entry.text.size();  ++k) {
      const auto& seq = s_escapes[uint8_t(entry.text[k])];
      if(ROCKET_EXPECT(seq[1] == 0)) {
        // Optimize the operation a little if it is a non-escaped character.
        log_text += seq[0];
      }
      else if(ROCKET_EXPECT(seq[0] != '\\')) {
        // Insert non-escaped characters verbatim.
        log_text += seq;
      }
      else {
        // Write an escaped sequence.
        do_color(log_text, conf, "7");  // reverse
        log_text += seq;
        do_color(log_text, conf, "27");  // reverse
      }
    }
    do_end_color(log_text, conf);
    log_text += "\n\v";

    // Write data to all streams.
    for(const auto& fd : strms) {
      // Note we only retry writing in case of EINTR.
      // `::write()` shall block, so partial writes are ignored.
  r:
      if(::write(fd, log_text.data(), log_text.size()) >= 0)
        continue;

      if(errno == EINTR)
        goto r;

      ::std::fprintf(stderr,
          "WARNING: Could not write log data: error %d: %m\n",
          errno);
    }
    return true;
  }

}  // namespace

POSEIDON_STATIC_CLASS_DEFINE(Async_Logger)
  {
    // constant data
    once_flag m_init_once;
    ::pthread_t m_thread = 0;

    // configuration
    mutable simple_mutex m_conf_mutex;
    Level_Config_Array m_conf_levels;

    // dynamic data
    mutable simple_mutex m_queue_mutex;
    condition_variable m_queue_avail;
    condition_variable m_queue_empty;
    ::std::deque<Log_Entry> m_queue;

    static void
    do_start()
      {
        self->m_init_once.call(
          [&] {
            // Create the thread. Note it is never joined or detached.
            simple_mutex::unique_lock lock(self->m_queue_mutex);
            self->m_thread = create_daemon_thread<do_thread_loop>("logger");
          });
      }

    static void
    do_thread_loop(void* /*param*/)
      {
        Log_Entry entry;
        size_t queue_size;

        // Await an entry and pop it.
        simple_mutex::unique_lock lock(self->m_queue_mutex);
        for(;;) {
          if(self->m_queue.empty()) {
            // Wait until an entry becomes available.
            self->m_queue_empty.notify_all();
            self->m_queue_avail.wait(lock);
            continue;
          }

          // Pop it.
          entry = ::std::move(self->m_queue.front());
          self->m_queue.pop_front();
          queue_size = self->m_queue.size();
          break;
        }
        lock.unlock();

        // Get configuration for this level.
        lock.lock(self->m_conf_mutex);
        const auto conf_levels = self->m_conf_levels;
        lock.unlock();

        // If there is congestion, discard trivial ones.
        if(conf_levels.at(entry.level).trivial && (queue_size >= 1024))
          return;

        // Write this entry.
        do_write_log_entry(conf_levels, ::std::move(entry));

        // Force a flush after the last entry.
        if(queue_size == 0)
          ::sync();
      }
  };

void
Async_Logger::
start()
  {
    self->do_start();
  }

void
Async_Logger::
reload()
  {
    // Load logger settings into this temporary array.
    const auto file = Main_Config::copy();
    Level_Config_Array levels;

    for(size_t k = 0;  k < levels.size();  ++k)
      do_load_level_config(levels[k], file, s_level_names[k].conf_name);

    // During destruction of `levels` the mutex should have been unlocked.
    // The swap operation is presumed to be fast, so we don't hold the
    // mutex for too long.
    simple_mutex::unique_lock lock(self->m_conf_mutex);
    self->m_conf_levels.swap(levels);
  }

bool
Async_Logger::
enabled(Log_Level level) noexcept
  {
    // Validate arguments.
    simple_mutex::unique_lock lock(self->m_conf_mutex);
    if(level >= self->m_conf_levels.size())
      return false;

    // The level is enabled if at least one stream is enabled.
    const auto& conf = self->m_conf_levels[level];
    return (conf.out_fd != -1) || conf.out_path.size();
  }

bool
Async_Logger::
enqueue(Log_Level level, const char* file, long line, const char* func, cow_string&& text)
  {
    // Get configuration for this level.
    simple_mutex::unique_lock lock(self->m_conf_mutex);
    const auto conf_levels = self->m_conf_levels;
    lock.unlock();

    // Compose the entry.
    Log_Entry entry;
    entry.level = level;
    entry.file = file;
    entry.line = line;
    entry.func = func;
    entry.text = ::std::move(text);

    entry.thr_lwpid = static_cast<::pid_t>(::syscall(__NR_gettid));
    ::stpcpy(entry.thr_name, "<unknown>");
    ::pthread_getname_np(::pthread_self(), entry.thr_name, sizeof(entry.thr_name));

    // If the logger thread has not been created, write it immediately.
    if(ROCKET_UNEXPECT(self->m_thread == 0))
      return do_write_log_entry(conf_levels, ::std::move(entry));

    // Push the entry.
    lock.lock(self->m_queue_mutex);
    self->m_queue.emplace_back(::std::move(entry));
    if(level <= log_level_error) {
      // Error messages should be written immediately.
      // We still have to maintain the order of all messages, so write them all.
      do {
        entry = ::std::move(self->m_queue.front());
        self->m_queue.pop_front();
        do_write_log_entry(conf_levels,::std::move(entry));
      }
      while(!self->m_queue.empty());

      // Force a flush after the last entry.
      ::sync();
    }
    else {
      // Push and forget.
      self->m_queue_avail.notify_one();
    }
    return true;
  }

void
Async_Logger::
synchronize(long msecs) noexcept
  {
    // Wait until the log queue becomes empty.
    simple_mutex::unique_lock lock(self->m_queue_mutex);
    self->m_queue_empty.wait_for(lock, msecs, [] { return self->m_queue.empty();  });
  }

}  // namespace poseidon
