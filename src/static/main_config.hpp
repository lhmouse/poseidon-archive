// This file is part of Poseidon.
// Copyleft 2022, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_STATIC_MAIN_CONFIG_
#define POSEIDON_STATIC_MAIN_CONFIG_

#include "../fwd.hpp"
#include "../core/config_file.hpp"

namespace poseidon {

// This class reads 'main.conf' and provides thread-safe accessors.
// Objects of this class are recommended to be static.
class Main_Config
  {
  private:
    mutable simple_mutex m_mutex;
    Config_File m_file;

  public:
    // Constructs an empty configuration file.
    Main_Config();

  public:
    ASTERIA_NONCOPYABLE_DESTRUCTOR(Main_Config);

    // Reloads 'main.conf'.
    // If this function fails, an exception is thrown, and there is no effect.
    // This function is thread-safe.
    void
    reload();

    // Copies the current file.
    // Configuration files are reference-counted and cheap to copy.
    // This function is thread-safe.
    Config_File
    copy() noexcept;
  };

}  // namespace poseidon

#endif
