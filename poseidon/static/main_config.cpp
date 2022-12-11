// This file is part of Poseidon.
// Copyleft 2022, LH_Mouse. All wrongs reserved.

#include "../precompiled.ipp"
#include "main_config.hpp"

namespace poseidon {

Main_Config::
Main_Config()
  {
  }

Main_Config::
~Main_Config()
  {
  }

void
Main_Config::
reload()
  {
    Config_File file;
    file.reload(::rocket::sref("main.conf"));

    // Set up new data.
    plain_mutex::unique_lock lock(this->m_mutex);
    this->m_file.swap(file);
  }

Config_File
Main_Config::
copy() noexcept
  {
    plain_mutex::unique_lock lock(this->m_mutex);
    return this->m_file;
  }

}  // namespace poseidon
