// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "async_logger.hpp"
#include "main_config.hpp"
#include "../core/config_file.hpp"
#include "../xutilities.hpp"

namespace poseidon {
namespace {

using Level = Async_Logger::Level;
using Entry = Async_Logger::Entry;

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

struct Level_Config
  {
    cow_string color;
    int out_fd = -1;
    cow_string out_path;
  };

using Level_Config_Array = array<Level_Config, ::rocket::countof(s_level_names)>;

void
do_load_level_config(Level_Config& conf, const Config_File& file, const char* name)
  {
    // Read color settings.
    // If we decide to disable color later on, we clear this string.
    if(const auto qcolor = file.get_string_opt({ "logger", "levels", name, "color" })) {
      // Set the color string.
      conf.color = ::std::move(*qcolor);
    }
    else {
      // Disable color.
      conf.color = ::rocket::sref("");
    }

    // Read stream settings.
    if(const auto qstrm = file.get_string_opt({ "logger", "levels", name, "stream" })) {
      // Set standard file descriptors.
      const auto strm = qstrm->safe_c_str();
      if(::strcmp(strm, "stdout") == 0) {
        conf.out_fd = STDOUT_FILENO;
      }
      else if(::strcmp(strm, "stderr") == 0) {
        conf.out_fd = STDERR_FILENO;
      }
      else {
        conf.out_fd = -1;
      }

      // Read the color setting for this stream.
      if(const auto qcolor = file.get_bool_opt({ "logger", "streams", strm, "color" })) {
        // Honor the user's specification.
        if(!*qcolor)
          conf.color = ::rocket::sref("");
      }
      else {
        // Disable color if the stream is not a terminal.
        if(!::isatty(conf.out_fd))
          conf.color = ::rocket::sref("");
      }

      // Read the alternative output file.
      if(const auto qfile = file.get_string_opt({ "logger", "streams", strm, "file" })) {
        // Set the output file path.
        conf.out_path = ::std::move(*qfile);
      }
      else {
        // Disable output file.
        conf.out_path = ::rocket::sref("");
      }
    }
    else {
      // Disable all outputs.
      conf.out_fd = -1;
      conf.out_path = ::rocket::sref("");
    }
  }

constexpr char s_escapes[][8] =
  {
    "\\0",    "\\x01",  "\\x02",  "\\x03",  "\\x04",  "\\x05",  "\\x06",  "\\a",
    "\\b",    "\\t",    "\\n",    "\\v",    "\\f",    "\\r",    "\\x0E",  "\\x0F",
    "\\x10",  "\\x11",  "\\x12",  "\\x13",  "\\x14",  "\\x15",  "\\x16",  "\\x17",
    "\\x18",  "\\x19",  "\\x1A",  "\\x1B",  "\\x1C",  "\\x1D",  "\\x1E",  "\\x1F",
    " ",      "!",      "\\\"",   "#",      "$",      "%",      "&",      "\\'",
    "(",      ")",      "*",      "+",      ",",      "-",      ".",      "/",
    "0",      "1",      "2",      "3",      "4",      "5",      "6",      "7",
    "8",      "9",      ":",      ";",      "<",      "=",      ">",      "?",
    "@",      "A" ,     "B",      "C" ,     "D",      "E",      "F" ,     "G",
    "H",      "I",      "J",      "K",      "L",      "M",      "N",      "O",
    "P",      "Q",      "R",      "S",      "T",      "U",      "V",      "W",
    "X",      "Y",      "Z",      "[",      "\\\\",   "]",      "^",      "_",
    "`",      "a",      "b",      "c",      "d",      "e",      "f",      "g",
    "h",      "i" ,     "j",      "k" ,     "l",      "m",      "n" ,     "o",
    "p",      "q",      "r",      "s",      "t",      "u",      "v",      "w",
    "x",      "y",      "z",      "{",      "|",      "}",      "~",      "\\x7F",
    "\x80",   "\x81",   "\x82",   "\x83",   "\x84",   "\x85",   "\x86",   "\x87",
    "\x88",   "\x89",   "\x8A",   "\x8B",   "\x8C",   "\x8D",   "\x8E",   "\x8F",
    "\x90",   "\x91",   "\x92",   "\x93",   "\x94",   "\x95",   "\x96",   "\x97",
    "\x98",   "\x99",   "\x9A",   "\x9B",   "\x9C",   "\x9D",   "\x9E",   "\x9F",
    "\xA0",   "\xA1",   "\xA2",   "\xA3",   "\xA4",   "\xA5",   "\xA6",   "\xA7",
    "\xA8",   "\xA9",   "\xAA",   "\xAB",   "\xAC",   "\xAD",   "\xAE",   "\xAF",
    "\xB0",   "\xB1",   "\xB2",   "\xB3",   "\xB4",   "\xB5",   "\xB6",   "\xB7",
    "\xB8",   "\xB9",   "\xBA",   "\xBB",   "\xBC",   "\xBD",   "\xBE",   "\xBF",
    "\\xC0",  "\\xC1",  "\xC2",   "\xC3",   "\xC4",   "\xC5",   "\xC6",   "\xC7",
    "\xC8",   "\xC9",   "\xCA",   "\xCB",   "\xCC",   "\xCD",   "\xCE",   "\xCF",
    "\xD0",   "\xD1",   "\xD2",   "\xD3",   "\xD4",   "\xD5",   "\xD6",   "\xD7",
    "\xD8",   "\xD9",   "\xDA",   "\xDB",   "\xDC",   "\xDD",   "\xDE",   "\xDF",
    "\xE0",   "\xE1",   "\xE2",   "\xE3",   "\xE4",   "\xE5",   "\xE6",   "\xE7",
    "\xE8",   "\xE9",   "\xEA",   "\xEB",   "\xEC",   "\xED",   "\xEE",   "\xEF",
    "\xF0",   "\xF1",   "\xF2",   "\xF3",   "\xF4",   "\\xF5",  "\\xF6",  "\\xF7",
    "\\xF8",  "\\xF9",  "\\xFA",  "\\xFB",  "\\xFC",  "\\xFD",  "\\xFE",  "\\xFF",
  };

template<typename SelfT>
bool
do_logger_loop(SelfT* self)
  {
    // Await an entry and pop it.
    ::rocket::mutex::unique_lock lock(self->m_queue.mutex);
    self->m_queue.avail.wait(lock, [&] { return self->m_queue.bpos != self->m_queue.epos;  });

    auto entry = ::std::move(self->m_queue.stor[static_cast<size_t>(self->m_queue.bpos)]);
    if(++(self->m_queue.bpos) == static_cast<ptrdiff_t>(self->m_queue.stor.size()))
      self->m_queue.bpos = 0;

    // Get configuration for this level.
    lock.assign(self->m_config.mutex);
    if(entry.level >= self->m_config.levels.size())
      return true;

    auto conf = self->m_config.levels[entry.level];
    const auto& names = s_level_names[entry.level];

    // Leave critical section.
    lock.unlock();

    // Get list of streams to write.
    sso_vector<::rocket::unique_posix_fd, 3> strms;

    if(conf.out_fd != -1)
      strms.emplace_back(conf.out_fd, nullptr);  // don't close

    if(conf.out_path.size()) {
      ::rocket::unique_posix_fd fd(::open(conf.out_path.c_str(),
                                          O_WRONLY | O_APPEND | O_CREAT, 0666),
                                   ::close);
      if(fd != -1)
        strms.emplace_back(::std::move(fd));
    }

    // If no stream is opened, return immediately.
    if(strms.empty())
      return true;

    // Compose the string to write.
    ::rocket::tinyfmt_str fmt;
    fmt.set_string(cow_string(1023, '/'));
    fmt.clear_string();

    // Write the timestamp.
    ::timespec ts;
    ::clock_gettime(CLOCK_REALTIME, &ts);
    ::tm tr;
    ::localtime_r(&(ts.tv_sec), &tr);

    if(conf.color.size()) {
      fmt << "\x1B[" << conf.color << "m";
    }
    // 'yyyy-mmmm-dd HH:MM:SS.sss'
    ::rocket::ascii_numput nump;
    fmt << nump.put_DU(static_cast<uint64_t>(tr.tm_year + 1900), 4);
    fmt << '-' << nump.put_DU(static_cast<uint64_t>(tr.tm_mon + 1), 2);
    fmt << '-' << nump.put_DU(static_cast<uint64_t>(tr.tm_mday), 2);
    fmt << ' ' << nump.put_DU(static_cast<uint64_t>(tr.tm_hour), 2);
    fmt << ':' << nump.put_DU(static_cast<uint64_t>(tr.tm_min), 2);
    fmt << ':' << nump.put_DU(static_cast<uint64_t>(tr.tm_sec), 2);
    fmt << '.' << nump.put_DU(static_cast<uint64_t>(ts.tv_nsec), 9);

    if(conf.color.size()) {
      fmt << "\x1B[0m";  // reset
    }
    fmt << ' ';

    // Write the log level string.
    if(conf.color.size()) {
      fmt << "\x1B[" << conf.color << ";7m";  // reverse
    }
    fmt << names.fmt_name;

    if(conf.color.size()) {
      fmt << "\x1B[0m";  // reset
    }
    fmt << ' ';

    // Write the thread ID and name.
    if(conf.color.size()) {
      fmt << "\x1B[30;1m";  // grey
    }
    fmt << "thread " << entry.thread.id << " [" << entry.thread.name << "]";

    if(conf.color.size()) {
      fmt << "\x1B[0m";  // reset
    }
    fmt << ' ';

    // Write the function name.
    if(conf.color.size()) {
      fmt << "\x1B[37;1m";  // bright white
    }
    fmt << "function `" << entry.func << "`";

    if(conf.color.size()) {
      fmt << "\x1B[0m";  // reset
    }
    fmt << ' ';

    // Write the file name and line number.
    if(conf.color.size()) {
      fmt << "\x1B[34;1m";  // bright blue
    }
    fmt << "@ " << entry.file << ':' << entry.line;

    if(conf.color.size()) {
      fmt << "\x1B[0m";  // reset
    }
    fmt << "\n\t";

    // Write the message.
    if(conf.color.size()) {
      fmt << "\x1B[" << conf.color << "m";
    }
    for(size_t k = 0;  k < entry.text.size();  ++k) {
      size_t ch = entry.text[k] & 0xFF;
      const auto& sq = s_escapes[ch];

      // Insert this escapd sequence.
      // Optimize the operation a little if it consists of only one character.
      if(ROCKET_EXPECT(sq[1] == 0))
        fmt << sq[0];
      else
        fmt << sq;
    }

    if(conf.color.size()) {
      fmt << "\x1B[0m";  // reset
    }
    fmt << '\n';

    const auto str = fmt.extract_string();

    // Write data to all streams.
    for(const auto& fd : strms) {
      const char* bp = str.data();
      const char* ep = str.data() + str.size();
      ::ssize_t nwritten;

      // Write the string, ignoring write errors.
      while(bp < ep) {
        nwritten = ::write(fd, bp, static_cast<size_t>(ep - bp));

        if((nwritten < 0) && (errno == EINTR))
          continue;  // retry

        if(nwritten <= 0)
          break;  // fail

        bp += nwritten;
      }
    }
    return true;
  }

}  // namespace

POSEIDON_STATIC_CLASS_DEFINE(Async_Logger)
  {
    struct
      {
        mutable ::rocket::mutex mutex;
        Level_Config_Array levels;
      }
      m_config;

    struct
      {
        mutable ::rocket::mutex mutex;
        ::rocket::condition_variable avail;

        // circular queue (if `bpos == epos` then empty)
        ptrdiff_t bpos = 0;
        ptrdiff_t epos = 0;
        std_vector<Entry> stor;
      }
      m_queue;

    opt<::pthread_t> m_thread;
  };

void
Async_Logger::
reload()
  {
    // Load logger settings into this temporary array.
    Level_Config_Array temp;
    auto file = Main_Config::copy();
    for(size_t k = 0;  k < temp.size();  ++k)
      do_load_level_config(temp[k], file, s_level_names[k].conf_name);

    // During destruction of `temp` the mutex should have been unlocked.
    // The swap operation is presumed to be fast, so we don't hold the mutex
    // for too long.
    ::rocket::mutex::unique_lock lock(self->m_config.mutex);
    self->m_config.levels.swap(temp);
  }

void
Async_Logger::
start()
  {
    if(self->m_thread)
      return;

    // Create the thread. Note it is never joined or detached.
    self->m_thread = create_daemon_thread<decltype(self), do_logger_loop>(self, "logger");
  }

bool
Async_Logger::
is_enabled(Level level)
noexcept
  {
    // Lock config for reading.
    ::rocket::mutex::unique_lock lock(self->m_config.mutex);

    // Validate arguments.
    if(level >= self->m_config.levels.size())
      return false;

    const auto& conf = self->m_config.levels[level];

    // The level is enabled if at least one stream is enabled.
    return (conf.out_fd != -1) || conf.out_path.size();
  }

void
Async_Logger::
write(Entry&& entry)
  {
    // Lock queue for modification.
    ::rocket::mutex::unique_lock lock(self->m_queue.mutex);

    // Get the number of unused elements in the queue.
    // Note the queue is empty if `bpos == epos`.do_logger_loop
    ptrdiff_t navail = self->m_queue.bpos - self->m_queue.epos;
    if(navail < 0)
      navail += static_cast<ptrdiff_t>(self->m_queue.stor.size());

    // Ensure the queue will not be full after pushing, otherwise it would be
    // impractical to tell whether the queue is empty or full when `bpos == epos`.
    constexpr ptrdiff_t nrsrv = 10;
    if(ROCKET_UNEXPECT(navail < nrsrv)) {
      // Insert elements at `epos`, pushing elements towards the back.
      self->m_queue.stor.insert(self->m_queue.stor.begin() + self->m_queue.epos,
                                nrsrv, { });

      // Update `bpos` as necessary, unless the queue is empty.
      if(self->m_queue.epos < self->m_queue.bpos)
        self->m_queue.bpos += nrsrv;
      navail += nrsrv;
    }

    // Push the element.
    self->m_queue.stor[static_cast<size_t>(self->m_queue.epos)] = ::std::move(entry);
    if(++(self->m_queue.epos) == static_cast<ptrdiff_t>(self->m_queue.stor.size()))
      self->m_queue.epos = 0;
    self->m_queue.avail.notify_one();
  }

}  // poseidon
