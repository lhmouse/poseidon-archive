// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_STATIC_MAIN_CONFIG_HPP_
#define POSEIDON_STATIC_MAIN_CONFIG_HPP_

#include "../fwd.hpp"
#include "../core/config_file.hpp"

namespace poseidon {

class Main_Config
  {
  private:
    ASTERIA_NONCOPYABLE_DESTRUCTOR(Main_Config)
      = delete;

  public:
    // Clears all data.
    // This function is thread-safe.
    static
    void
    clear()
    noexcept;

    // Reloads the file `etc/poseidon/main.conf'.
    // If this function fails, an exception is thrown, and there is no effect.
    // This function is thread-safe.
    static
    void
    reload();

    // Obtains a copy the global config file.
    // Because config files are copy-on-write, this function never fails.
    // This function is thread-safe.
    static
    Config_File
    copy()
    noexcept;
  };

}  // namespace poseidon

#endif
