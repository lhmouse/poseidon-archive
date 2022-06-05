// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_STATIC_ASYNC_LOGGER_
#define POSEIDON_STATIC_ASYNC_LOGGER_

#include "../fwd.hpp"

namespace poseidon {

class Async_Logger
  {
    POSEIDON_STATIC_CLASS_DECLARE(Async_Logger);

  public:
    // Reloads settings from main config.
    // If this function fails, an exception is thrown, and there is no effect.
    // This function is thread-safe.
    static
    void
    reload();

    // Checks whether a given level of log has any output streams.
    // This function is thread-safe.
    ROCKET_CONST static
    bool
    enabled(Log_Level level) noexcept;

    // Enqueues a log entry and returns the total number of entries that are pending.
    // If this function fails, an exception is thrown, and there is no effect.
    // This function is thread-safe.
    static
    bool
    enqueue(Log_Level level, const char* file, long line, const char* func, cow_string&& text);

    // Waits until all pending log entries are delivered to output devices.
    // This function is thread-safe.
    static
    void
    synchronize() noexcept;
  };

// Composes a string and submits it to the logger.
#define POSEIDON_LOG_ENQUEUE(LEVEL, ...)  \
    (::poseidon::Async_Logger::enabled(::poseidon::log_level_##LEVEL)  \
      && ([&](void) -> bool {  \
        try {  \
          ::poseidon::Async_Logger::enqueue(  \
              ::poseidon::log_level_##LEVEL, __FILE__, __LINE__, __func__,  \
              ::asteria::format_string("" __VA_ARGS__));  \
        }  \
        catch(::std::exception& ispxfUAu) {  \
          ::fprintf(stderr,  \
              "%s: %s\n", __func__, ispxfUAu.what());  \
        }  \
        return true;  \
      }()))

#define POSEIDON_LOG_FATAL(...)   POSEIDON_LOG_ENQUEUE(fatal,  __VA_ARGS__)
#define POSEIDON_LOG_ERROR(...)   POSEIDON_LOG_ENQUEUE(error,  __VA_ARGS__)
#define POSEIDON_LOG_WARN(...)    POSEIDON_LOG_ENQUEUE(warn,   __VA_ARGS__)
#define POSEIDON_LOG_INFO(...)    POSEIDON_LOG_ENQUEUE(info,   __VA_ARGS__)
#define POSEIDON_LOG_DEBUG(...)   POSEIDON_LOG_ENQUEUE(debug,  __VA_ARGS__)
#define POSEIDON_LOG_TRACE(...)   POSEIDON_LOG_ENQUEUE(trace,  __VA_ARGS__)

}  // namespace poseidon

#endif
