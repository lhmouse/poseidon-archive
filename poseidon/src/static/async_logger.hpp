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
    // Note each level has a hardcoded name and number.
    // Don't change their values or reorder them.
    enum Level : uint8_t
      {
        level_fatal  = 0,
        level_error  = 1,
        level_warn   = 2,
        level_info   = 3,
        level_debug  = 4,
        level_trace  = 5,
      };

    struct Entry
      {
        Level level;

        const char* file;
        long line;
        const char* func;

        struct
          {
            long id;
            char name[16];
          }
          thread;

        cow_string text;
      };

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
    is_enabled(Level level)
    noexcept;

    // Writes a line of log.
    // If this function fails, an exception is thrown, and there is no effect.
    // This function is thread-safe.
    static
    void
    write(Entry&& entry);
  };

}  // namespace poseidon

#endif
