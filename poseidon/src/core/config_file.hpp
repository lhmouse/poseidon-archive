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
    Config_File()
    noexcept
      { }

    explicit
    Config_File(const char* path)
      { this->reload(path);  }

    ASTERIA_COPYABLE_DESTRUCTOR(Config_File);

  public:
    const cow_string&
    abspath()
    const noexcept
      { return this->m_abspath;  }

    bool
    empty()
    const noexcept
      { return this->m_root.empty();  }

    ::asteria::V_object::const_iterator
    begin()
    const noexcept
      { return this->m_root.begin();  }

    ::asteria::V_object::const_iterator
    end()
    const noexcept
      { return this->m_root.end();  }

    const ::asteria::V_object&
    root()
    const noexcept
      { return this->m_root;  }

    Config_File&
    clear()
    noexcept
      {
        this->m_root.clear();
        return *this;
      }

    Config_File&
    swap(Config_File& other)
    noexcept
      {
        this->m_root.swap(other.m_root);
        return *this;
      }

    // This function loads the file denoted by `path`.
    // If this function fails, an exception is thrown and there is no
    // effect (strong exception safety guarantee).
    Config_File&
    reload(const char* path);

    
  };

inline
void
swap(Config_File& lhs, Config_File& rhs)
noexcept
  { lhs.swap(rhs);  }

}  // namespace poseidon

#endif
