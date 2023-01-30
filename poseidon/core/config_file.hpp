// This file is part of Poseidon.
// Copyleft 2022, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_CORE_CONFIG_FILE_
#define POSEIDON_CORE_CONFIG_FILE_

#include "../fwd.hpp"
#include <asteria/value.hpp>

namespace poseidon {

// This class can be used to parse a configuration file.
// Please see 'etc/poseidon/main.default.conf' for details about the file
// format and syntax.
class Config_File
  {
  private:
    cow_string m_path;
    ::asteria::V_object m_root;

  public:
    // Constructs an empty file.
    constexpr
    Config_File() noexcept = default;

    // Loads the file denoted by `path`, like `reload(path)`.
    explicit
    Config_File(const cow_string& path);

  public:
    ASTERIA_COPYABLE_DESTRUCTOR(Config_File);

    // Returns the absolute file path.
    // If no file has been loaded, an empty string is returned.
    cow_string
    path() const noexcept
      { return this->m_path;  }

    // Accesses the contents.
    ::asteria::V_object
    root() const noexcept
      { return this->m_root;  }

    bool
    empty() const noexcept
      { return this->m_root.empty();  }

    Config_File&
    clear() noexcept
      {
        this->m_path.clear();
        this->m_root.clear();
        return *this;
      }

    Config_File&
    swap(Config_File& other) noexcept
      {
        this->m_path.swap(other.m_path);
        this->m_root.swap(other.m_root);
        return *this;
      }

    // Loads the file denoted by `path`.
    // This function provides strong exception guarantee. In case of failure,
    // an exception is thrown, and the contents of this object are unchanged.
    Config_File&
    reload(const cow_string& file_path);

    // Gets a value denoted by a path, which shall not be empty.
    // If the path does not denote an existent value, a statically allocated
    // null value is returned. If during path resolution, an attempt is made
    // to get a field of a non-object, an exception is thrown.
    const ::asteria::Value&
    query(initializer_list<phsh_string> value_path) const;

    template<typename... SegmentsT>
    const ::asteria::Value&
    query(const SegmentsT&... value_path) const
      { return this->query({ ::rocket::sref(value_path)... });  }
  };

inline
void
swap(Config_File& lhs, Config_File& rhs) noexcept
  {
    lhs.swap(rhs);
  }

}  // namespace poseidon
#endif
