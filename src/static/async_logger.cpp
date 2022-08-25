// This file is part of Poseidon.
// Copyleft 2022, LH_Mouse. All wrongs reserved.

#include "../precompiled.ipp"
#include "async_logger.hpp"
#include "../core/config_file.hpp"
#include "../utils.hpp"
#include <sys/syscall.h>
#include <time.h>

namespace poseidon {
namespace {

struct Level_Config
  {
    char tag[8] = "";
    cow_string color;
    int stdio = -1;
    cow_string file;
    bool trivial = false;
  };

constexpr char escapes[][5] =
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

void
do_load_level_config(Level_Config& lconf, const Config_File& file, const char* name)
  {
    // Set the tag.
    ::memcpy(lconf.tag, "[     ] ", 8);
    for(size_t k = 0;  (k < 5) && name[k];  ++k)
      lconf.tag[k + 1] = ::rocket::ascii_to_upper(name[k]);

    // Read the color code sequence of the level.
    auto value = file.query("logger", name, "color");
    if(value.is_string())
      lconf.color = value.as_string();
    else if(!value.is_null())
      POSEIDON_LOG_WARN((
          "Ignoring `logger.$1.color`: expecting a `string`, got `$2`",
          "[in configuration file '$3']"),
          name, value, file.path());

    // Read the output standard stream.
    cow_string str;
    value = file.query("logger", name, "stdio");
    if(value.is_string())
      str = value.as_string();
    else if(!value.is_null())
      POSEIDON_LOG_WARN((
          "Ignoring `logger.$1.stdio`: expecting a `string`, got `$2`",
          "[in configuration file '$3']"),
          name, value, file.path());

    if(str == "stdout")
      lconf.stdio = STDOUT_FILENO;
    else if(str == "stderr")
      lconf.stdio = STDERR_FILENO;
    else
      POSEIDON_LOG_WARN((
          "Ignoring `logger.$1.stdio`: invalid standard stream name `$2`",
          "[in configuration file '$3']"),
          name, str, file.path());

    // Read the output file path.
    value = file.query("logger", name, "file");
    if(value.is_string())
      lconf.file = value.as_string();
    else if(!value.is_null())
      POSEIDON_LOG_WARN((
          "Ignoring `logger.$1.file`: expecting a `string`, got `$2`",
          "[in configuration file '$3']"),
          name, value, file.path());

    // Read verbosity settings.
    value = file.query("logger", name, "trivial");
    if(value.is_boolean())
      lconf.trivial = value.as_boolean();
    else if(!value.is_null())
      POSEIDON_LOG_WARN((
          "Ignoring `logger.$1.trivial`: expecting a `boolean`, got `$2`",
          "[in configuration file '$3']"),
          name, value, file.path());
  }

inline
void
do_color(cow_string& data, const Level_Config& lconf, const char* code)
  {
    if(!lconf.color.empty())
      data << "\x1B[" << code << "m";
  }

inline
bool
do_write_loop(int fd, const cow_string& data) noexcept
  {
    for(;;)
      if(::write(fd, data.data(), data.size()) >= 0)
        return true;
      else if(errno != EINTR)
        return false;
  }

void
do_write_nothrow(const Level_Config& lconf, const Async_Logger::Queued_Message& msg) noexcept
  try {
    // Compose the string to write.
    cow_string data;
    data.reserve(2047);

    // Write the timestamp and tag for sorting.
    do_color(data, lconf, lconf.color.c_str());

    ::timespec ts;
    ::clock_gettime(CLOCK_REALTIME, &ts);
    ::tm tr;
    ::localtime_r(&(ts.tv_sec), &tr);

    ::rocket::ascii_numput nump;
    uint64_t datetime = (uint32_t) tr.tm_year + 1900;
    datetime *= 100;
    datetime += (uint32_t) tr.tm_mon + 1;
    datetime *= 100;
    datetime += (uint32_t) tr.tm_mday;
    datetime *= 100;
    datetime += (uint32_t) tr.tm_hour;
    datetime *= 100;
    datetime += (uint32_t) tr.tm_min;
    datetime *= 100;
    datetime += (uint32_t) tr.tm_sec;
    nump.put_DU(datetime);

    data.append(nump.data() +  0, 4);
    data.push_back('-');
    data.append(nump.data() +  4, 2);
    data.push_back('-');
    data.append(nump.data() +  6, 2);
    data.push_back(' ');
    data.append(nump.data() +  8, 2);
    data.push_back(':');
    data.append(nump.data() + 10, 2);
    data.push_back(':');
    data.append(nump.data() + 12, 2);

    nump.put_DU((uint32_t) ts.tv_nsec, 9);
    data.push_back('.');
    data.append(nump.data(), 9);
    data.push_back(' ');

    do_color(data, lconf, "7");  // inverse
    data.append(lconf.tag, 7);
    do_color(data, lconf, "0");  // reset
    data += " ";

    // Write the thread name and ID.
    do_color(data, lconf, "30;1");  // grey
    data += "THREAD ";
    nump.put_DU(msg.thrd_lwpid);
    data.append(nump.data(), nump.size());
    data += " \"";
    data += msg.thrd_name;
    data += "\" ";

    // Write the function name.
    do_color(data, lconf, "37;1");  // bright white
    data += "FUNCTION `";
    data += msg.func;
    data += "` ";

    // Write the source file name and line number.
    do_color(data, lconf, "34;1");  // bright blue
    data += "SOURCE \'";
    data += msg.file;
    data += ':';
    nump.put_DU(msg.line);
    data.append(nump.data(), nump.size());
    data += "\'\n";

    // Write the message.
    do_color(data, lconf, "0");  // reset
    do_color(data, lconf, lconf.color.c_str());
    data += '\t';

    for(char ch : msg.text) {
      const char* seq = escapes[(uint8_t) ch];
      if(seq[1] == 0) {
        // non-escaped
        data.push_back(seq[0]);
      }
      else if(seq[0] == '\\') {
        // non-printable or bad
        do_color(data, lconf, "7");
        data.append(seq);
        do_color(data, lconf, "27");
      }
      else
        data.append(seq);
    }

    // Remove trailing space characters.
    size_t pos = data.find_last_not_of(" \f\n\r\t\v");
    data.erase(pos + 1);
    data += "\n\v";
    do_color(data, lconf, "0");  // reset

    // Write text to streams. Errors are ignored.
    if(!lconf.file.empty()) {
      unique_posix_fd ofd(::open(lconf.file.c_str(), O_WRONLY | O_APPEND | O_CREAT, 0644));
      if(ofd)
        do_write_loop(ofd, data);
      else
        ::fprintf(stderr,
            "WARNING: Could not open log file '%s' for appending: %m\n",
            lconf.file.c_str());
    }

    if(lconf.stdio != -1)
      do_write_loop(lconf.stdio, data);
  }
  catch(exception& stdex) {
    ::fprintf(stderr,
        "WARNING: Failed to write log text: %s\n"
        "[exception class `%s`]\n",
        stdex.what(), typeid(stdex).name());
  }

}  // namespace

POSEIDON_HIDDEN_STRUCT(Async_Logger, Level_Config);

Async_Logger::
Async_Logger()
  {
  }

Async_Logger::
~Async_Logger()
  {
  }

void
Async_Logger::
reload(const Config_File& file)
  {
    // Parse new configuration.
    cow_vector<Level_Config> levels(6);
    uint32_t level_mask = 0;

    do_load_level_config(levels.mut(log_level_trace), file, "trace");
    do_load_level_config(levels.mut(log_level_debug), file, "debug");
    do_load_level_config(levels.mut(log_level_info ), file, "info" );
    do_load_level_config(levels.mut(log_level_warn ), file, "warn" );
    do_load_level_config(levels.mut(log_level_error), file, "error");
    do_load_level_config(levels.mut(log_level_fatal), file, "fatal");

    for(size_t k = 0;  k != levels.size();  ++k)
      if((levels[k].stdio != -1) || !levels[k].file.empty())
        level_mask |= 1U << k;

    if(level_mask == 0)
      ::fprintf(stderr, "WARNING: Logger disabled\n");

    // Set up new data.
    plain_mutex::unique_lock lock(this->m_conf_mutex);
    this->m_conf_levels.swap(levels);
    this->m_conf_level_mask.store(level_mask);
  }

void
Async_Logger::
thread_loop()
  {
    // Get all pending elements.
    plain_mutex::unique_lock lock(this->m_queue_mutex);
    while(this->m_queue.empty())
      this->m_queue_avail.wait(lock);

    recursive_mutex::unique_lock io_sync_lock(this->m_io_mutex);
    this->m_io_queue.clear();
    this->m_io_queue.swap(this->m_queue);
    lock.unlock();

    // Get configuration.
    lock.lock(this->m_conf_mutex);
    const auto levels = this->m_conf_levels;
    lock.unlock();

    // Write all elements.
    for(const auto& msg : this->m_io_queue)
      if(msg.level < levels.size())
        if((this->m_io_queue.size() <= 1024U) || !levels[msg.level].trivial)
          do_write_nothrow(levels[msg.level], msg);

    this->m_io_queue.clear();
    io_sync_lock.unlock();
    ::sync();
  }

void
Async_Logger::
enqueue(Queued_Message&& msg)
  {
    // Fill in the name and LWP ID of the calling thread.
    ::strncpy(msg.thrd_name, "[unknown]", sizeof(msg.thrd_name));
    ::pthread_getname_np(::pthread_self(), msg.thrd_name, sizeof(msg.thrd_name));
    msg.thrd_lwpid = (uint32_t) ::syscall(__NR_gettid);

    // Enqueue the element.
    plain_mutex::unique_lock lock(this->m_queue_mutex);
    this->m_queue.emplace_back(::std::move(msg));
    this->m_queue_avail.notify_one();
  }

void
Async_Logger::
synchronize() noexcept
  {
    // Get all pending elements.
    plain_mutex::unique_lock lock(this->m_queue_mutex);
    recursive_mutex::unique_lock io_sync_lock(this->m_io_mutex);
    if(this->m_queue.empty())
      return;

    this->m_io_queue.clear();
    this->m_io_queue.swap(this->m_queue);
    lock.unlock();

    // Get configuration.
    lock.lock(this->m_conf_mutex);
    const auto levels = this->m_conf_levels;
    lock.unlock();

    // Write all elements.
    for(const auto& msg : this->m_io_queue)
      if(msg.level < levels.size())
        do_write_nothrow(levels[msg.level], msg);

    this->m_io_queue.clear();
    io_sync_lock.unlock();
    ::sync();
  }

}  // namespace poseidon
