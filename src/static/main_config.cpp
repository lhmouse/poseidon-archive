// This file is part of Poseidon.
// Copyleft 2022, LH_Mouse. All wrongs reserved.

#include "../precompiled.ipp"
#include "main_config.hpp"

namespace poseidon {
namespace {

// This is the path to the main configuration file, relative to
// the working directory.
constexpr auto file_path = ::rocket::sref("main.conf");

}  // namespace

Main_Config::
Main_Config()
  {
    this->m_file.reload(file_path);
  }

Main_Config::
~Main_Config()
  {
  }

void
Main_Config::
reload()
  {
    Config_File new_file;
    new_file.reload(file_path);

    // Set up new data.
    simple_mutex::unique_lock lock(this->m_mutex);
    this->m_file.swap(new_file);
  }

Config_File
Main_Config::
copy() noexcept
  {
    simple_mutex::unique_lock lock(this->m_mutex);
    return this->m_file;
  }

}  // namespace poseidon
