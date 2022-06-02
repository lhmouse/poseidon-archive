// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_UTILS_
#  error Please include <poseidon/utils.hpp> instead.
#endif

namespace poseidon {
namespace details_utils {

template<typename... ParamsT>
ROCKET_NEVER_INLINE static
void
format_log(Log_Level level, const char* file, long line, const char* func, const char* templ, const ParamsT&... params) noexcept
  try {
    // Compose the message.
    ::rocket::tinyfmt_str fmt;
    format(fmt, templ, params...);  // ADL intended
    auto text = fmt.extract_string();

    // Push a new log entry.
    Async_Logger::enqueue(level, file, line, func, ::std::move(text));
  }
  catch(exception& stdex) {
    // Ignore this exception, but print a message.
    ::std::fprintf(stderr,
        "%s: %s\n"
        "[exception class `%s` thrown from '%s:%ld']\n",
        func, stdex.what(),
        typeid(stdex).name(), file, line);
  }

template<typename... ParamsT>
[[noreturn]] ROCKET_NEVER_INLINE static
void
format_throw(const char* file, long line, const char* func, const char* templ, const ParamsT&... params)
  {
    // Compose the message.
    ::rocket::tinyfmt_str fmt;
    format(fmt, templ, params...);  // ADL intended
    auto text = fmt.extract_string();

    // Push a new log entry.
    static constexpr auto level = log_level_warn;
    if(Async_Logger::enabled(level))
      Async_Logger::enqueue(level, file, line, func, "POSEIDON_THROW: " + text);

    // Throw the exception.
    ::rocket::sprintf_and_throw<::std::runtime_error>(
        "%s: %s\n"
        "[thrown from '%s:%ld']",
        func, text.c_str(),
        file, line);
  }

}  // namespace details_utils
}  // namespace poseidon
