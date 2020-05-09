// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "main_config.hpp"
#include "../utilities.hpp"

namespace poseidon {
namespace {

::std::mutex s_mutex;
Config_File s_conf;

}  // namespace

void
Main_Config::
clear()
noexcept
  {
    // Create an empty file.
    Config_File temp;

    // During destruction of `temp` the mutex should have been unlocked.
    // The swap operation is presumed to be fast, so we don't hold the mutex
    // for too long.
    ::std::lock_guard<::std::mutex> lock(s_mutex);
    s_conf.swap(temp);
  }

void
Main_Config::
reload()
  {
    // Load the global config file by relative path.
    // An exception is thrown upon failure.
    Config_File temp("etc/poseidon/main.conf");

    // During destruction of `temp` the mutex should have been unlocked.
    // The swap operation is presumed to be fast, so we don't hold the mutex
    // for too long.
    ::std::lock_guard<::std::mutex> lock(s_mutex);
    s_conf.swap(temp);
  }

Config_File
Main_Config::
copy()
noexcept
  {
    // Note again config files are copy-on-write and cheap to copy.
    ::std::lock_guard<::std::mutex> lock(s_mutex);
    return s_conf;
  }

}  // poseidon
