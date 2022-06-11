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
    int fd = -1;
    cow_string color;
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

    // Read level settings.
    cow_string stream;
    int color = -1;

    auto value = file.query("logger", "levels", name, "stream");
    if(value.is_string())
      stream = value.as_string();
    else if(!value.is_null())
      POSEIDON_LOG_WARN((
          "Ignoring `logger.levels.$1.stream`: expecting `string`, got `$2`",
          "[in configuration file '$3']"),
          name, ::asteria::describe_type(value.type()), file.path());

    if(! stream.empty()) {
      // Set special streams.
      if(stream == ::rocket::sref("stdout"))
        lconf.fd = STDOUT_FILENO;
      else if(stream == ::rocket::sref("stderr"))
        lconf.fd = STDERR_FILENO;

      // Read color settings of the stream. If no explicit `true` or `false` is
      // given, color is enabled if the file descriptor denotes a terminal.
      value = file.query("logger", "streams", stream, "color");
      if(value.is_boolean())
        color = value.as_boolean();
      else if(!value.is_null())
        POSEIDON_LOG_WARN((
            "Ignoring `logger.streams.$1.color`: expecting `boolean`, got `$2`",
            "[in configuration file '$3']"),
            stream, ::asteria::describe_type(value.type()), file.path());

      if(color < 0)
        color = (lconf.fd != -1) && ::isatty(lconf.fd);

      if(color) {
        // Read the color code sequence of the level.
        value = file.query("logger", "levels", name, "color");
        if(value.is_string())
          lconf.color = value.as_string();
        else if(!value.is_null())
          POSEIDON_LOG_WARN((
              "Ignoring `logger.levels.$1.color`: expecting `string`, got `$2`",
              "[in configuration file '$3']"),
              name, ::asteria::describe_type(value.type()), file.path());
      }

      // Read the output file path.
      value = file.query("logger", "streams", stream, "file");
      if(value.is_string())
        lconf.file = value.as_string();
      else if(!value.is_null())
        POSEIDON_LOG_WARN((
            "Ignoring `logger.streams.$1.file`: expecting `string`, got `$2`",
            "[in configuration file '$3']"),
            stream, ::asteria::describe_type(value.type()), file.path());
    }

    // Read verbosity settings.
    value = file.query("logger", "levels", name, "trivial");
    if(value.is_boolean())
      lconf.trivial = value.as_boolean();
    else if(!value.is_null())
      POSEIDON_LOG_WARN((
          "Ignoring `logger.levels.$1.trivial`: expecting `boolean`, got `$2`",
          "[in configuration file '$3']"),
          name, ::asteria::describe_type(value.type()), file.path());
  }

inline
void
do_color(cow_string& data, const Level_Config& lconf, const char* code)
  {
    if(! lconf.color.empty())
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
do_write_nothrow(const Level_Config& lconf, const Async_Logger::Element& elem) noexcept
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
    nump.put_DU(elem.thrd_lwpid);
    data.append(nump.data(), nump.size());
    data += " \"";
    data += elem.thrd_name;
    data += "\" ";

    // Write the function name.
    do_color(data, lconf, "37;1");  // bright white
    data += "FUNCTION `";
    data += elem.func;
    data += "` ";

    // Write the source file name and line number.
    do_color(data, lconf, "34;1");  // bright blue
    data += "SOURCE \'";
    data += elem.file;
    data += ':';
    nump.put_DU(elem.line);
    data.append(nump.data(), nump.size());
    data += "\'\n";

    // Write the message.
    do_color(data, lconf, "0");  // reset
    do_color(data, lconf, lconf.color.c_str());
    data += '\t';

    for(char ch : elem.text) {
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
    if(! lconf.file.empty()) {
      ::rocket::unique_posix_fd ofd(::close);
      if(ofd.reset(::open(lconf.file.c_str(), O_WRONLY | O_APPEND | O_CREAT, 0644)))
        do_write_loop(ofd, data);
      else
        ::fprintf(stderr,
            "WARNING: Could not open log file '%s' for appending: %m\n",
             lconf.file.c_str());
    }

    if(lconf.fd != -1)
      do_write_loop(lconf.fd, data);
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
thread_loop()
  {
    // Get all pending elements.
    plain_mutex::unique_lock lock(this->m_queue_mutex);
    while(this->m_queue.empty())
      this->m_queue_avail.wait(lock);

    plain_mutex::unique_lock io_lock(this->m_io_mutex);
    this->m_io_queue.clear();
    this->m_io_queue.swap(this->m_queue);
    lock.unlock();

    // Get configuration.
    lock.lock(this->m_conf_mutex);
    const auto conf = this->m_conf;
    lock.unlock();

    // Write all elements.
    for(const auto& elem : this->m_io_queue)
      if(elem.level < conf.size())
        if(! conf[elem.level].trivial || (this->m_io_queue.size() <= 1024U))
          do_write_nothrow(conf[elem.level], elem);

    this->m_io_queue.clear();
    io_lock.unlock();
    ::sync();
  }

void
Async_Logger::
reload(const Config_File& file)
  {
    // Parse new configuration.
    vector<Level_Config> conf(6);
    uint32_t mask = 0;

    do_load_level_config(conf.at(log_level_trace), file, "trace");
    do_load_level_config(conf.at(log_level_debug), file, "debug");
    do_load_level_config(conf.at(log_level_info ), file, "info" );
    do_load_level_config(conf.at(log_level_warn ), file, "warn" );
    do_load_level_config(conf.at(log_level_error), file, "error");
    do_load_level_config(conf.at(log_level_fatal), file, "fatal");

    for(size_t k = 0;  k != conf.size();  ++k)
      if((conf[k].fd != -1) || ! conf[k].file.empty())
        mask |= 1U << k;

    // Set up new data.
    plain_mutex::unique_lock lock(this->m_conf_mutex);
    this->m_conf.swap(conf);
    this->m_mask.store(mask);
  }

void
Async_Logger::
enqueue(Element&& elem)
  {
    // Fill in the name and LWP ID of the calling thread.
    ::strncpy(elem.thrd_name, "[unknown]", sizeof(elem.thrd_name));
    ::pthread_getname_np(::pthread_self(), elem.thrd_name, sizeof(elem.thrd_name));
    elem.thrd_lwpid = (uint32_t) ::syscall(__NR_gettid);

    // Enqueue the element.
    plain_mutex::unique_lock lock(this->m_queue_mutex);
    this->m_queue.emplace_back(::std::move(elem));
    this->m_queue_avail.notify_one();
  }

void
Async_Logger::
synchronize() noexcept
  {
    // Get all pending elements.
    plain_mutex::unique_lock lock(this->m_queue_mutex);
    if(this->m_queue.empty())
      return;

    plain_mutex::unique_lock io_lock(this->m_io_mutex);
    this->m_io_queue.clear();
    this->m_io_queue.swap(this->m_queue);
    lock.unlock();

    // Get configuration.
    lock.lock(this->m_conf_mutex);
    const auto conf = this->m_conf;
    lock.unlock();

    // Write all elements.
    for(const auto& elem : this->m_io_queue)
      if(elem.level < conf.size())
        do_write_nothrow(conf[elem.level], elem);

    this->m_io_queue.clear();
    io_lock.unlock();
    ::sync();
  }

}  // namespace poseidon
