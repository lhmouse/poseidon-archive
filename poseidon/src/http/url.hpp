// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_HTTP_URL_HPP_
#define POSEIDON_HTTP_URL_HPP_

#include "../fwd.hpp"

namespace poseidon {

class URL
  {
  private:
    /* http://user@example.com:80/path/to/document?foo=1&bar=2#location
     * ~~~~   ~~~~ ~~~~~~~~~~~ ~~ ~~~~~~~~~~~~~~~~ ~~~~~~~~~~~ ~~~~~~~~
     *    \      \ /           /  /                /           /
     * scheme    | |           |  |                |           |
     * userinfo -+ |           |  |                |           |
     * host     ---+           |  |                |           |
     * port     ---------------+  |                |           |
     * path     ------------------+                |           |
     * query    -----------------------------------+           |
     * fragment -----------------------------------------------+
    **/
    cow_string m_scheme;
    cow_string m_userinfo;
    cow_string m_host;
    opt<uint16_t> m_port;
    cow_string m_path;
    cow_string m_raw_query;
    cow_string m_raw_fragment;

  public:
    constexpr
    URL() noexcept
      = default;

    explicit
    URL(const cow_string& str)
      { this->parse(str);  }

    URL&
    operator=(const cow_string& str)
      { return this->parse(str);  }

  public:
    ASTERIA_COPYABLE_DESTRUCTOR(URL);

    // Gets the scheme.
    const cow_string&
    scheme() noexcept
      { return this->m_scheme;  }

    // Sets the scheme.
    // If the scheme is non-empty, this function ensures that it really
    // conforms to RFC 3986 and converts letters into lowercase.
    // An exception is thrown if the scheme is invalid.
    URL&
    clear_scheme() noexcept
      { return this->m_scheme.clear(), *this;  }

    URL&
    set_scheme(const cow_string& val);

    // Gets the user information.
    const cow_string&
    userinfo() noexcept
      { return this->m_userinfo;  }

    // Sets the user information, which may comprise arbitrary characters.
    URL&
    clear_userinfo() noexcept
      { return this->m_userinfo.clear(), *this;  }

    URL&
    set_userinfo(const cow_string& val);

    // Gets the host name.
    const cow_string&
    host() noexcept
      { return this->m_host;  }

    // Sets the host name.
    // The host name may be an IP address enclosed in a pair of brackets.
    // But it may comprise arbitrary characters otherwise.
    // An exception is thrown if the host name is not a valid IP address.
    URL&
    clear_host() noexcept
      { return this->m_host.clear(), *this;  }

    URL&
    set_host(const cow_string& val);

    // Gets the port.
    // If the port field is absent, a default one is chosen according to
    // the scheme. If no default port is available, zero is returned.
    ROCKET_PURE_FUNCTION
    uint16_t
    default_port() const noexcept;

    uint16_t
    port() const noexcept
      { return this->m_port ? *(this->m_port) : this->default_port();  }

    // Sets the port.
    URL&
    clear_port() noexcept
      { return this->m_port.reset(), *this;  }

    URL&
    set_port(uint16_t val);

    // Gets the path.
    // The slash initiator is not included.
    const cow_string&
    path() const noexcept
      { return this->m_path;  }

    // Sets the path.
    // A path may comprise arbitrary characters.
    URL&
    clear_path() noexcept
      { return this->m_path.clear(), *this;  }

    URL&
    set_path(const cow_string& val);

    // Gets the percent-encoded query string.
    const cow_string&
    raw_query() const noexcept
      { return this->m_raw_query;  }

    // Sets the query string.
    // The query string is pasted to the URL intact, so it cannot contain unsafe
    // characters. It must really conform to RFC 3986. An exception is thrown if
    // the query string is invalid.
    URL&
    clear_raw_query() noexcept
      { return this->m_raw_query.clear(), *this;  }

    URL&
    set_raw_query(const cow_string& val);

    // Gets the percent-encoded fragment.
    const cow_string&
    raw_fragment() const noexcept
      { return this->m_raw_fragment;  }

    // Sets the fragment stirng.
    // The fragment string is pasted to the URL intact, so it cannot contain
    // unsafe characters. It must really conform to RFC 3986. An exception is
    // thrown if the fragment string is invalid.
    URL&
    clear_raw_fragment() noexcept
      { return this->m_raw_fragment.clear(), *this;  }

    URL&
    set_raw_fragment(const cow_string& val);

    // These are general modifiers.
    URL&
    clear() noexcept
      {
        this->m_scheme.clear();
        this->m_userinfo.clear();
        this->m_host.clear();
        this->m_port.reset();
        this->m_path.clear();
        this->m_raw_query.clear();
        this->m_raw_fragment.clear();
        return *this;
      }

    URL&
    swap(URL& other) noexcept
      {
        this->m_scheme.swap(other.m_scheme);
        this->m_userinfo.swap(other.m_userinfo);
        this->m_host.swap(other.m_host);
        this->m_port.swap(other.m_port);
        this->m_path.swap(other.m_path);
        this->m_raw_query.swap(other.m_raw_query);
        this->m_raw_fragment.swap(other.m_raw_fragment);
        return *this;
      }

    // Converts this URL to a string.
    // This is the inverse function of `parse()`.
    tinyfmt&
    print(tinyfmt& fmt) const;

    // Parses a URL string.
    // An exception is thrown if the URL is invalid.
    URL&
    parse(const cow_string& str);
  };

inline
void
swap(URL& lhs, URL& rhs) noexcept
  { lhs.swap(rhs);  }

inline
tinyfmt&
operator<<(tinyfmt& fmt, const URL& url)
  { return url.print(fmt);  }

}  // namespace poseidon

#endif
