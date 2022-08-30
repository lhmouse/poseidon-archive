// This file is part of Poseidon.
// Copyleft 2022, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_HTTP_FIELDS_
#define POSEIDON_HTTP_FIELDS_

#include "../fwd.hpp"

namespace poseidon {

// This class can be used to store the headers of a HTTP message, or arguments
// decoded from the query part of a URL.
class HTTP_Fields
  {
  public:
    using container_type  = cow_vector<pair<cow_string, cow_string>>;
    using value_type      = container_type::value_type;
    using allocator_type  = container_type::allocator_type;
    using const_iterator  = container_type::const_iterator;
    using iterator        = container_type::iterator;

    using const_reverse_iterator  = container_type::const_reverse_iterator;
    using reverse_iterator        = container_type::reverse_iterator;

  private:
    container_type m_stor;

  public:
    // Contructs an empty container.
    constexpr
    HTTP_Fields() noexcept = default;

  public:
    ASTERIA_COPYABLE_DESTRUCTOR(HTTP_Fields);

    // These are general accessors.
    bool
    empty() const noexcept
      { return this->m_stor.empty();  }

    size_t
    capacity() const noexcept
      { return this->m_stor.capacity();  }

    size_t
    size() const noexcept
      { return this->m_stor.size();  }

    ptrdiff_t
    ssize() const noexcept
      { return this->m_stor.ssize();  }

    const_iterator
    begin() const noexcept
      { return this->m_stor.begin();  }

    const_iterator
    end() const noexcept
      { return this->m_stor.end();  }

    const_reverse_iterator
    rbegin() const noexcept
      { return this->m_stor.rbegin();  }

    const_reverse_iterator
    rend() const noexcept
      { return this->m_stor.rend();  }

    iterator
    mut_begin() noexcept
      { return this->m_stor.mut_begin();  }

    iterator
    mut_end() noexcept
      { return this->m_stor.mut_end();  }

    reverse_iterator
    mut_rbegin() noexcept
      { return this->m_stor.mut_rbegin();  }

    reverse_iterator
    mut_rend() noexcept
      { return this->m_stor.mut_rend();  }

    const value_type&
    at(size_t index) const
      { return this->m_stor.at(index);  }

    const cow_string&
    name(size_t index) const
      { return this->m_stor.at(index).first;  }

    const cow_string&
    value(size_t index) const
      { return this->m_stor.at(index).second;  }

    value_type&
    mut(size_t index)
      { return this->m_stor.mut(index);  }

    cow_string&
    mut_name(size_t index)
      { return this->m_stor.mut(index).first;  }

    cow_string&
    mut_value(size_t index)
      { return this->m_stor.mut(index).second;  }

    HTTP_Fields&
    swap(HTTP_Fields& other) noexcept
      {
        this->m_stor.swap(other.m_stor);
        return *this;
      }

    HTTP_Fields&
    reserve(size_t res_arg);

    HTTP_Fields&
    shrink_to_fit();

    HTTP_Fields&
    clear() noexcept;

    iterator
    insert(const_iterator pos, const value_type& field);

    iterator
    insert(const_iterator pos, value_type&& field);

    iterator
    insert(const_iterator pos, const cow_string& name, const cow_string& value);

    iterator
    erase(const_iterator pos);

    iterator
    erase(const_iterator first, const_iterator last);

    // Prints the contents of this container to a stream, as a list of HTTP
    // headers. This function is suspected to produce human-readable results for
    // debugging purposes. In order to produce valid HTTP headers, all names and
    // values shall be well-formed.
    tinyfmt&
    print(tinyfmt& fmt) const;

    cow_string
    print_to_string() const;

    // Decodes HTTP headers from a string. Empty lines are ignored.
    // If `false` is returned or an exception is thrown, the contents of
    // this object are unspecified.
    bool
    parse(const cow_string& lines);

    // Encodes the contents of this container as an option in a single HTTP
    // header. Fields are written like `name=value; name=value`, where `value`
    // is quoted if necessary. Empty values are omitted as well as their equals
    // signs.
    tinyfmt&
    options_encode(tinyfmt& fmt) const;

    cow_string
    options_encode_as_string() const;

    // Decodes an option. Empty fields are ignored.
    // If `false` is returned or an exception is thrown, the contents of
    // this object are unspecified.
    bool
    options_decode(const cow_string& text);

    // Encodes the contents of this container in percent-encoding, according to
    // RFC 2396. This form is used for arguments in HTTP requests.
    tinyfmt&
    query_encode(tinyfmt& fmt) const;

    cow_string
    query_encode_as_string() const;

    // Decodes arguments from a query.
    // If `false` is returned or an exception is thrown, the contents of
    // this object are unspecified.
    bool
    query_decode(const cow_string& text);

    // Gets the last field with the given name. Names are case-insensitive.
    // If no such field exists, a null pointer is returned.
    const value_type*
    find_opt(const cow_string& name) const noexcept;

    value_type*
    mut_find_opt(const cow_string& name) noexcept;

    // Squashes all fields with the given name. Names are case-insensitive.
    // If no such field exists, a null pointer is returned. Otherwise, they
    // are joined with commas and a pointer to the result field is returned.
    value_type*
    squash_opt(const cow_string& name);

    // Appends a new field.
    value_type&
    append(const value_type& field);

    value_type&
    append(value_type&& field);

    value_type&
    append(const cow_string& name, const cow_string& value);

    value_type&
    append_empty(const cow_string& name);

    // Removes all fields with the given name. Names are case-insensitive.
    // The number of fields that have been removed is returned.
    size_t
    erase(const cow_string& name);
  };

inline
void
swap(HTTP_Fields& lhs, HTTP_Fields& rhs) noexcept
  {
    lhs.swap(rhs);
  }

inline
tinyfmt&
operator<<(tinyfmt& fmt, const HTTP_Fields& fields)
  {
    return fields.print(fmt);
  }

}  // namespace poseidon

#endif
