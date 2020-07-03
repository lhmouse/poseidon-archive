// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_CORE_URL_HPP_
#define POSEIDON_CORE_URL_HPP_

#include "../fwd.hpp"

namespace poseidon {

class URL
  {
  private:
    /*            http://user@example.com:80/path/to/document?foo=1&bar=2#location
     *            ^~~~   ^~~~ ^~~~~~~~~~~ ^~ ^~~~~~~~~~~~~~~~ ^~~~~~~~~~~ ^~~~~~~~
     * scheme   --/      |    |           |  |                |           |
     * userinfo ---------/    |           |  |                |           |
     * host     --------------/           |  |                |           |
     * port     --------------------------/  |                |           |
     * path     -----------------------------/                |           |
     * query    ----------------------------------------------/           |
     * fragment ----------------------------------------------------------/
    **/
    cow_string m_scheme;
    cow_string m_userinfo;
    cow_string m_host;
    opt<uint16_t> m_port;
    cow_string m_path;
    cow_string m_query;
    cow_string m_fragment;

  public:
    constexpr
    URL()
    noexcept
      { }

    explicit
    URL(const cow_string& str)
      { this->parse(str);  }

    URL&
    operator=(const cow_string& str)
      { return this->parse(str);  }

    ASTERIA_COPYABLE_DESTRUCTOR(URL);

  private:
    uint16_t
    do_get_default_port()
    const noexcept;

    cow_string&
    do_verify_and_set_scheme(cow_string&& val);

    cow_string&
    do_verify_and_set_host(cow_string&& val);

    cow_string&
    do_verify_and_set_query(cow_string&& val);

  public:
    // Gets the scheme.
    const cow_string&
    scheme()
    noexcept
      { return this->m_scheme;  }

    // Sets the scheme.
    // If the scheme is non-empty, this function ensures that it really
    // conforms to RFC 3986 and converts letters into lowercase.
    // An exception is thrown if the scheme is invalid.
    URL&
    set_scheme(cow_string val)
      { return this->do_verify_and_set_scheme(::std::move(val)), *this;  }

    URL&
    clear_scheme()
    noexcept
      { return this->m_scheme.clear(), *this;  }

    // Gets the user information.
    const cow_string&
    userinfo()
    noexcept
      { return this->m_userinfo;  }

    // Sets the user information, which may comprise arbitrary characters.
    URL&
    set_userinfo(cow_string val)
      { return this->m_userinfo = ::std::move(val), *this;  }

    URL&
    clear_userinfo()
    noexcept
      { return this->m_userinfo.clear(), *this;  }

    // Gets the host name.
    const cow_string&
    host()
    noexcept
      { return this->m_host;  }

    // Sets the host name.
    // The host name may be an IP address enclosed in a pair of brackets.
    // But it may comprise arbitrary characters otherwise.
    // An exception is thrown if the host name is not a valid IP address.
    URL&
    set_host(cow_string val)
      { return this->do_verify_and_set_host(::std::move(val)), *this;  }

    URL&
    clear_host()
    noexcept
      { return this->m_host.clear(), *this;  }

    // Gets the port.
    // If the port field is absent, a default one is chosen according to
    // the scheme. If no default port is available, zero is returned.
    uint16_t
    port()
    const noexcept
      { return this->m_port ? *(this->m_port) : this->do_get_default_port();  }

    // Sets the port.
    URL&
    set_port(uint16_t val)
    noexcept
      { return this->m_port = val, *this;  }

    URL&
    clear_port()
    noexcept
      { return this->m_port.reset(), *this;  }

    // Gets the path.
    // The slash initiator is not included.
    const cow_string&
    path()
    const noexcept
      { return this->m_path;  }

    // Sets the path.
    // A path may comprise arbitrary characters.
    URL&
    set_path(cow_string val)
    noexcept
      { return this->m_path = ::std::move(val), *this;  }

    URL&
    clear_path()
    noexcept
      { return this->m_path.clear(), *this;  }

    // Gets the query string.
    const cow_string&
    query()
    const noexcept
      { return this->m_query;  }

    // Sets the query string.
    // The query string is pasted to the URL intact, so it cannot contain unsafe
    // characters. This function ensures that it really conforms to RFC 3986.
    // An exception is thrown if the query string is invalid.
    URL&
    set_query(cow_string val)
      { return this->do_verify_and_set_query(::std::move(val)), *this;  }

    URL&
    clear_query()
    noexcept
      { return this->m_query.clear(), *this;  }

    // Gets the fragment.
    const cow_string&
    fragment()
    const noexcept
      { return this->m_fragment;  }

    // Sets the fragment.
    // A fragment may comprise arbitrary characters.
    URL&
    set_fragment(cow_string val)
    noexcept
      { return this->m_fragment = ::std::move(val), *this;  }

    URL&
    clear_fragment()
    noexcept
      { return this->m_fragment.clear(), *this;  }

    // These are general modifiers.
    URL&
    clear()
    noexcept
      {
        this->m_scheme.clear();
        this->m_userinfo.clear();
        this->m_host.clear();
        this->m_port.reset();
        this->m_path.clear();
        this->m_query.clear();
        this->m_fragment.clear();
        return *this;
      }

    URL&
    swap(URL& other)
    noexcept
      {
        this->m_scheme.swap(other.m_scheme);
        this->m_userinfo.swap(other.m_userinfo);
        this->m_host.swap(other.m_host);
        this->m_port.swap(other.m_port);
        this->m_path.swap(other.m_path);
        this->m_query.swap(other.m_query);
        this->m_fragment.swap(other.m_fragment);
        return *this;
      }

    // Converts this URL to a string.
    // This is the inverse function of `parse()`.
    tinyfmt&
    print(tinyfmt& fmt)
    const;

    // Parses a URL from a string.
    // An exception is thrown if the URL is invalid.
    URL&
    parse(const cow_string& str);
  };

inline
void
swap(URL& lhs, URL& rhs)
noexcept
  { lhs.swap(rhs);  }

inline
tinyfmt&
operator<<(tinyfmt& fmt, const URL& url)
  { return url.print(fmt);  }

}  // namespace poseidon

#endif
