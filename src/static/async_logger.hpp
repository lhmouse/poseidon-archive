// This file is part of Poseidon.
// Copyleft 2022, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_STATIC_ASYNC_LOGGER_
#define POSEIDON_STATIC_ASYNC_LOGGER_

#include "../fwd.hpp"

namespace poseidon {

// This class buffers log messages and write them asynchronously.
// Objects of this class are recommended to be static.
class Async_Logger
  {
  public:
    // This represents a queued log message.
    struct Element
      {
        Log_Level level;
        const char* file;
        size_t line;
        const char* func;
        cow_string text;

        // Users should not set these.
        char thrd_name[16];
        uint32_t thrd_lwpid;
      };

  private:
    mutable plain_mutex m_conf_mutex;
    struct Level_Config;
    vector<Level_Config> m_conf;
    atomic_relaxed<uint32_t> m_mask;

    mutable plain_mutex m_queue_mutex;
    vector<Element> m_queue;
    condition_variable m_queue_avail;

    mutable plain_mutex m_io_mutex;
    vector<Element> m_io_queue;

  public:
    // Creates a logger that outputs to nowhere.
    Async_Logger();

  public:
    ASTERIA_NONCOPYABLE_DESTRUCTOR(Async_Logger);

    // Performs I/O operation.
    // This function should be called by the logger thread repeatedly.
    void
    thread_loop();

    // Reloads configuration from 'main.conf'.
    // If this function fails, an exception is thrown, and there is no effect.
    // This function is thread-safe.
    void
    reload(const Config_File& file);

    // Checks whether a given level is enabled.
    // This function is thread-safe.
    ROCKET_PURE
    bool
    enabled(Log_Level level) const noexcept
      {
        return (level < 16U) && (this->m_mask.load() >> level & 1U);
      }

    // Enqueues a log message.
    // If this function fails, an exception is thrown, and there is no effect.
    // This function is thread-safe.
    void
    enqueue(Element&& elem);

    // Waits until all pending log entries are delivered to output devices.
    // This function is thread-safe.
    void
    synchronize() noexcept;
  };

}  // namespace poseidon

#endif
