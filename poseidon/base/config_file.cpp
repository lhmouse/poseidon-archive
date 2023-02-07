// This file is part of Poseidon.
// Copyleft 2022 - 2023, LH_Mouse. All wrongs reserved.

#include "../precompiled.ipp"
#include "config_file.hpp"
#include "../utils.hpp"
#include <stdlib.h>
#include <asteria/library/system.hpp>

namespace poseidon {

Config_File::
Config_File(const cow_string& path)
  {
    this->reload(path);
  }

Config_File::
~Config_File()
  {
  }

Config_File&
Config_File::
reload(const cow_string& file_path)
  {
    // Resolve the path to an absolute one.
    ::rocket::unique_ptr<char, void (void*)> abs_path(::free);
    if(!abs_path.reset(::realpath(file_path.safe_c_str(), nullptr)))
      POSEIDON_THROW((
          "Could not resolve path to configuration file '$2'",
          "[`realpath()` failed: $1]"),
          format_errno(), file_path);

    // Read the file.
    cow_string path(abs_path.get());
    ::asteria::V_object root = ::asteria::std_system_conf_load_file(path);

    // Set new contents. This shall not throw exceptions.
    this->m_path = ::std::move(path);
    this->m_root = ::std::move(root);
    return *this;
  }

const ::asteria::Value&
Config_File::
query(initializer_list<phsh_string> value_path) const
  {
    // We would like to return a `Value`, so the path shall not be empty.
    auto pcur = value_path.begin();
    if(pcur == value_path.end())
      POSEIDON_THROW(("Empty value path not valid"));

    // Resolve the first segment.
    auto parent = &(this->m_root);
    auto value = parent->ptr(*pcur);

    // Resolve all remaining segments.
    while(value && (++pcur != value_path.end())) {
      if(value->is_null())
        return ::asteria::null_value;

      if(!value->is_object()) {
        // Fail.
        cow_string vpstr;
        auto pbak = value_path.begin();
        vpstr << pbak->rdstr();
        while(++pbak != pcur)
          vpstr << '.' << pbak->rdstr();

        POSEIDON_THROW((
            "Unexpected type of `$1` (expecting an `object`, got `$2`)",
            "[in configuration file '$3']"),
            vpstr, *value, this->m_path);
      }

      // Descend into this child object.
      parent = &(value->as_object());
      value = parent->ptr(*pcur);
    }

    // If the path does not exist, return the static null value.
    return value ? *value : ::asteria::null_value;
  }

}  // namespace poseidon
