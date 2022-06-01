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
    ROCKET_PURE static
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

}  // namespace poseidon

#endif
