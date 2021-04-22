// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_CORE_CONFIG_FILE_HPP_
#define POSEIDON_CORE_CONFIG_FILE_HPP_

#include "../fwd.hpp"
#include <asteria/value.hpp>

namespace poseidon {

class Config_File
  {
  private:
    cow_string m_abspath;
    ::asteria::V_object m_root;

  public:
    explicit constexpr
    Config_File() noexcept
      = default;

    explicit
    Config_File(const cow_string& path)
      { this->reload(path);  }

  public:
    ASTERIA_COPYABLE_DESTRUCTOR(Config_File);

    const cow_string&
    abspath() const noexcept
      { return this->m_abspath;  }

    bool
    empty() const noexcept
      { return this->m_root.empty();  }

    ::asteria::V_object::const_iterator
    begin() const noexcept
      { return this->m_root.begin();  }

    ::asteria::V_object::const_iterator
    end() const noexcept
      { return this->m_root.end();  }

    const ::asteria::V_object&
    root() const noexcept
      { return this->m_root;  }

    Config_File&
    clear() noexcept
      {
        this->m_root.clear();
        return *this;
      }

    Config_File&
    swap(Config_File& other) noexcept
      {
        this->m_abspath.swap(other.m_abspath);
        this->m_root.swap(other.m_root);
        return *this;
      }

    // Loads the file denoted by `path`.
    // If this function fails, an exception is thrown and the contents are
    // indeterminate (basic exception safety guarantee).
    Config_File&
    reload(const cow_string& path);

    // Gets a value denoted by a path, which shall not be empty.
    // If the path does not denote an existent value, a statically allocated
    // null value is returned. If during path resolution, an attempt is made
    // to get a field of a value which is not an object, an exception is thrown.
    const ::asteria::Value&
    get_value(const char* const* psegs, size_t nsegs) const;

    const ::asteria::Value&
    get_value(initializer_list<const char*> path) const
      { return this->get_value(path.begin(), path.size());  }

    // These functions behave like `query_value()` except that they perform
    // type checking and conversion as needed.
    opt<bool>
    get_bool_opt(const char* const* psegs, size_t nsegs) const;

    opt<bool>
    get_bool_opt(initializer_list<const char*> path) const
      { return this->get_bool_opt(path.begin(), path.size());  }

    opt<int64_t>
    get_int64_opt(const char* const* psegs, size_t nsegs) const;

    opt<int64_t>
    get_int64_opt(initializer_list<const char*> path) const
      { return this->get_int64_opt(path.begin(), path.size());  }

    opt<double>
    get_double_opt(const char* const* psegs, size_t nsegs) const;

    opt<double>
    get_double_opt(initializer_list<const char*> path) const
      { return this->get_double_opt(path.begin(), path.size());  }

    opt<cow_string>
    get_string_opt(const char* const* psegs, size_t nsegs) const;

    opt<cow_string>
    get_string_opt(initializer_list<const char*> path) const
      { return this->get_string_opt(path.begin(), path.size());  }

    opt<::asteria::V_array>
    get_array_opt(const char* const* psegs, size_t nsegs) const;

    opt<::asteria::V_array>
    get_array_opt(initializer_list<const char*> path) const
      { return this->get_array_opt(path.begin(), path.size());  }

    opt<::asteria::V_object>
    get_object_opt(const char* const* psegs, size_t nsegs) const;

    opt<::asteria::V_object>
    get_object_opt(initializer_list<const char*> path) const
      { return this->get_object_opt(path.begin(), path.size());  }
  };

inline void
swap(Config_File& lhs, Config_File& rhs) noexcept
  { lhs.swap(rhs);  }

}  // namespace poseidon

#endif
