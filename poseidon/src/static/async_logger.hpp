// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_STATIC_ASYNC_LOGGER_HPP_
#define POSEIDON_STATIC_ASYNC_LOGGER_HPP_

#include "../fwd.hpp"

namespace poseidon {

class Async_Logger
  {
    POSEIDON_STATIC_CLASS_DECLARE(Async_Logger);

  public:
    // Creates the logger thread if one hasn't been created.
    static
    void
    start();

    // Reloads settings from main config.
    // If this function fails, an exception is thrown, and there is no effect.
    // This function is thread-safe.
    static
    void
    reload();

    // Checks whether a given level of log has any output streams.
    // This function is thread-safe.
    ROCKET_PURE_FUNCTION
    static
    bool
    is_enabled(Log_Level level)
    noexcept;

    // Retrieves the number of log entries that are pending.
    // This function is thread-safe.
    ROCKET_PURE_FUNCTION
    static
    size_t
    queue_size()
    noexcept;

    // Enqueues a log entry and returns the total number of entries that are pending.
    // If this function fails, an exception is thrown, and there is no effect.
    // This function is thread-safe.
    static
    bool
    enqueue(Log_Level level, const char* file, long line, const char* func,
            cow_string&& text);
  };

}  // namespace poseidon

#endif
