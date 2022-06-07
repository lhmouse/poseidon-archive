// This file is part of Poseidon.
// Copyleft 2022, LH_Mouse. All wrongs reserved.

#include "../precompiled.ipp"
#include "config_file.hpp"
#include "../utils.hpp"
#include <stdlib.h>
#include <asteria/library/system.hpp>

namespace poseidon {

Config_File::
Config_File() noexcept
  {
  }

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
reload(const cow_string& path)
  {
    // Resolve the path to an absolute one.
    ::rocket::unique_ptr<char, void (void*)> upath(::realpath(path.safe_c_str(), nullptr), ::free);
    if(!upath)
      POSEIDON_THROW(
          "Could not resolve path '$2'\n"
          "[`realpath()` failed: $1]",
          format_errno(), path);

    // Read the file.
    cow_string path_new(upath.get());
    ::asteria::V_object root_new = ::asteria::std_system_conf_load_file(path_new);

    // Set new contents. This shall not throw exceptions.
    this->m_path = ::std::move(path_new);
    this->m_root = ::std::move(root_new);
    return *this;
  }

const ::asteria::Value&
Config_File::
query(const char* const* psegs, size_t nsegs) const
  {
    // We would like to return a `Value`, so the path shall not be empty.
    if(nsegs == 0)
      POSEIDON_THROW("Empty path not valid");

    // Resolve the first segment.
    auto parent = &(this->m_root);
    auto value = parent->ptr(::rocket::sref(psegs[0]));

    // Resolve all remaining segments.
    for(size_t k = 1;  value && (k != nsegs);  ++k) {
      if(value->is_null())
        return ::asteria::null_value;

      if(!value->is_object()) {
        // Fail.
        cow_string str;
        str << psegs[0];
        for(size_t r = 1;  r != k;  ++r)
          str << '.' << psegs[r];

        POSEIDON_THROW(
            "Unexpected type of `$2` (expecting `object`, got `$3`)\n"
            "[in configuration file '$1']",
            this->m_path, str, ::asteria::describe_type(value->type()));
      }

      // Descend into this child object.
      parent = &(value->as_object());
      value = parent->ptr(::rocket::sref(psegs[k]));
    }

    // If the path does not exist, return the static null value.
    return value ? *value : ::asteria::null_value;
  }

}  // namespace poseidon
