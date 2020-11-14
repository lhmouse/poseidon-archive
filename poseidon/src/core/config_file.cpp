// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "config_file.hpp"
#include "../util.hpp"
#include <asteria/library/system.hpp>

namespace poseidon {
namespace {

[[noreturn]]
void
do_throw_type_mismatch(const cow_string& abspath, const char* const* bptr,
                       size_t epos, const char* expect,
                       const ::asteria::Value& value)
  {
    cow_string path;
    path << '`';
    ::std::for_each(bptr, bptr + epos,
             [&](const char* s) { path << s << '.';  });
    path.mut_back() = '`';

    POSEIDON_THROW("Unexpected type of $1 (expecting $2, got `$3`)\n"
                   "[in file '$4']",
                   path, expect, ::asteria::describe_type(value.type()),
                   abspath);
  }

}  // namespace

Config_File::
~Config_File()
  {
  }

Config_File&
Config_File::
reload(const cow_string& path)
  {
    this->m_root = ::asteria::std_system_conf_load_file(path);
    return *this;
  }

const ::asteria::Value&
Config_File::
get_value(const char* const* psegs, size_t nsegs)
  const
  {
    auto qobj = &(this->m_root);
    size_t icur = 0;
    if(icur == nsegs)
      POSEIDON_THROW("Empty path not valid");

    for(;;) {
      // Find the child denoted by `*sptr`.
      // Return null if no such child exists or if an explicit null
      // is found.
      auto qchild = qobj->ptr(::rocket::sref(psegs[icur]));
      if(!qchild || qchild->is_null())
        return ::asteria::null_value;

      // Advance to the next segment.
      // If the end of `path` is reached, we are done.
      if(++icur == nsegs)
        return *qchild;

      // If more segments follow, the child must be an object.
      if(!qchild->is_object())
        do_throw_type_mismatch(this->m_abspath, psegs, icur,
                               "`object`", *qchild);

      qobj = &(qchild->as_object());
    }
  }

opt<bool>
Config_File::
get_bool_opt(const char* const* psegs, size_t nsegs)
  const
  {
    const auto& value = this->get_value(psegs, nsegs);
    if(value.is_null())
      return nullopt;

    if(!value.is_boolean())
      do_throw_type_mismatch(this->m_abspath, psegs, nsegs,
                             "`boolean`", value);

    return value.as_boolean();
  }

opt<int64_t>
Config_File::
get_int64_opt(const char* const* psegs, size_t nsegs)
  const
  {
    const auto& value = this->get_value(psegs, nsegs);
    if(value.is_null())
      return nullopt;

    if(!value.is_integer())
      do_throw_type_mismatch(this->m_abspath, psegs, nsegs,
                             "`integer`", value);

    return value.as_integer();
  }

opt<double>
Config_File::
get_double_opt(const char* const* psegs, size_t nsegs)
  const
  {
    const auto& value = this->get_value(psegs, nsegs);
    if(value.is_null())
      return nullopt;

    if(!value.is_convertible_to_real())
      do_throw_type_mismatch(this->m_abspath, psegs, nsegs,
                             "`integer` or `real`", value);

    return value.convert_to_real();
  }

opt<cow_string>
Config_File::
get_string_opt(const char* const* psegs, size_t nsegs)
  const
  {
    const auto& value = this->get_value(psegs, nsegs);
    if(value.is_null())
      return nullopt;

    if(!value.is_string())
      do_throw_type_mismatch(this->m_abspath, psegs, nsegs,
                             "`string`", value);

    return value.as_string();
  }

opt<::asteria::V_array>
Config_File::
get_array_opt(const char* const* psegs, size_t nsegs)
  const
  {
    const auto& value = this->get_value(psegs, nsegs);
    if(value.is_null())
      return nullopt;

    if(!value.is_array())
      do_throw_type_mismatch(this->m_abspath, psegs, nsegs,
                             "`array`", value);

    return value.as_array();
  }

opt<::asteria::V_object>
Config_File::
get_object_opt(const char* const* psegs, size_t nsegs)
  const
  {
    const auto& value = this->get_value(psegs, nsegs);
    if(value.is_null())
      return nullopt;

    if(!value.is_object())
      do_throw_type_mismatch(this->m_abspath, psegs, nsegs,
                             "`object`", value);

    return value.as_object();
  }

}  // namespace poseidon
