// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "config_file.hpp"
#include "../utils.hpp"
#include <asteria/library/system.hpp>

namespace poseidon {
namespace {

struct Implode
  {
    const char* const* psegs;
    size_t nsegs;
  };

constexpr Implode
do_implode(const char* const* p, size_t n) noexcept
  { return { p, n };  }

tinyfmt&
operator<<(tinyfmt& fmt, const Implode& imp)
  {
    if(imp.nsegs != 0) {
      fmt << imp.psegs[0];
      for(size_t k = 1;  k < imp.nsegs;  ++k)
        fmt << '.' << imp.psegs[k];
    }
    return fmt;
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
get_value(const char* const* psegs, size_t nsegs) const
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
      if(!qchild) {
        POSEIDON_LOG_DEBUG(
            "Undefined value `$2`\n"
            "[in configuration file '$1']",
            this->m_abspath, do_implode(psegs, nsegs));

        return ::asteria::null_value;
      }
      else if(qchild->is_null())
        return ::asteria::null_value;

      // Advance to the next segment.
      // If the end of `path` is reached, we are done.
      if(++icur == nsegs)
        return *qchild;

      // If more segments follow, the child must be an object.
      if(!qchild->is_object())
        POSEIDON_THROW(
            "Unexpected type of `$2` (expecting `object`, got `$3`)\n"
            "[in configuration file '$1']",
            this->m_abspath, do_implode(psegs, nsegs),
            ::asteria::describe_type(qchild->type()));

      qobj = &(qchild->as_object());
    }
  }

opt<bool>
Config_File::
get_bool_opt(const char* const* psegs, size_t nsegs) const
  {
    const auto& value = this->get_value(psegs, nsegs);
    if(value.is_null())
      return nullopt;

    if(!value.is_boolean())
      POSEIDON_THROW(
          "Unexpected type of `$2` (expecting `boolean`, got `$3`)\n"
          "[in configuration file '$1']",
          this->m_abspath, do_implode(psegs, nsegs),
          ::asteria::describe_type(value.type()));

    return value.as_boolean();
  }

opt<int64_t>
Config_File::
get_int64_opt(const char* const* psegs, size_t nsegs) const
  {
    const auto& value = this->get_value(psegs, nsegs);
    if(value.is_null())
      return nullopt;

    if(!value.is_integer())
      POSEIDON_THROW(
          "Unexpected type of `$2` (expecting `integer`, got `$3`)\n"
          "[in configuration file '$1']",
          this->m_abspath, do_implode(psegs, nsegs),
          ::asteria::describe_type(value.type()));

    return value.as_integer();
  }

opt<double>
Config_File::
get_double_opt(const char* const* psegs, size_t nsegs) const
  {
    const auto& value = this->get_value(psegs, nsegs);
    if(value.is_null())
      return nullopt;

    if(!value.is_real())
      POSEIDON_THROW(
          "Unexpected type of `$2` (expecting `number`, got `$3`)\n"
          "[in configuration file '$1']",
          this->m_abspath, do_implode(psegs, nsegs),
          ::asteria::describe_type(value.type()));

    return value.as_real();
  }

opt<cow_string>
Config_File::
get_string_opt(const char* const* psegs, size_t nsegs) const
  {
    const auto& value = this->get_value(psegs, nsegs);
    if(value.is_null())
      return nullopt;

    if(!value.is_string())
      POSEIDON_THROW(
          "Unexpected type of `$2` (expecting `string`, got `$3`)\n"
          "[in configuration file '$1']",
          this->m_abspath, do_implode(psegs, nsegs),
          ::asteria::describe_type(value.type()));

    return value.as_string();
  }

opt<::asteria::V_array>
Config_File::
get_array_opt(const char* const* psegs, size_t nsegs) const
  {
    const auto& value = this->get_value(psegs, nsegs);
    if(value.is_null())
      return nullopt;

    if(!value.is_array())
      POSEIDON_THROW(
          "Unexpected type of `$2` (expecting `array`, got `$3`)\n"
          "[in configuration file '$1']",
          this->m_abspath, do_implode(psegs, nsegs),
          ::asteria::describe_type(value.type()));

    return value.as_array();
  }

opt<::asteria::V_object>
Config_File::
get_object_opt(const char* const* psegs, size_t nsegs) const
  {
    const auto& value = this->get_value(psegs, nsegs);
    if(value.is_null())
      return nullopt;

    if(!value.is_object())
      POSEIDON_THROW(
          "Unexpected type of `$2` (expecting `object`, got `$3`)\n"
          "[in configuration file '$1']",
          this->m_abspath, do_implode(psegs, nsegs),
          ::asteria::describe_type(value.type()));

    return value.as_object();
  }

}  // namespace poseidon
