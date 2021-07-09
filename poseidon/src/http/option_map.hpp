// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_HTTP_OPTION_MAP_HPP_
#define POSEIDON_HTTP_OPTION_MAP_HPP_

#include "../fwd.hpp"
#include "../details/option_map.ipp"

namespace poseidon {

class Option_Map
  {
  public:
    using const_iterator          = details_option_map::Iterator<const cow_string>;
    using iterator                = details_option_map::Iterator<cow_string>;
    using const_reverse_iterator  = ::std::reverse_iterator<const_iterator>;
    using reverse_iterator        = ::std::reverse_iterator<iterator>;

  private:
    ::rocket::cow_vector<details_option_map::Bucket> m_stor;
    size_t m_nbkt = 0;

  public:
    explicit constexpr
    Option_Map() noexcept
      = default;

  private:
    ROCKET_PURE pair<const cow_string*, const cow_string*>
    do_range_hint(cow_string::shallow_type key, size_t hval) const noexcept;

    pair<cow_string*, cow_string*>
    do_mut_range_hint(cow_string::shallow_type key, size_t hval);

    size_t
    do_erase_hint(cow_string::shallow_type key, size_t hval);

    inline details_option_map::Bucket&
    do_reserve(const cow_string& key, size_t hval);

    cow_string&
    do_open_hint(const cow_string& key, size_t hval);

    cow_string&
    do_append_hint(const cow_string& key, size_t hval);

  public:
    ASTERIA_COPYABLE_DESTRUCTOR(Option_Map);

    // iterators
    const_iterator
    begin() const noexcept
      { return const_iterator(this->m_stor.data(), 0, this->m_stor.size());  }

    const_iterator
    end() const noexcept
      { return const_iterator(this->m_stor.data(),
                              this->m_stor.size(), this->m_stor.size());  }

    const_reverse_iterator
    rbegin() const noexcept
      { return const_reverse_iterator(this->end());  }

    const_reverse_iterator
    rend() const noexcept
      { return const_reverse_iterator(this->begin());  }

    iterator
    mut_begin()
      { return iterator(this->m_stor.mut_data(), 0, this->m_stor.size());  }

    iterator
    mut_end()
      { return iterator(this->m_stor.mut_data(),
                        this->m_stor.size(), this->m_stor.size());  }

    reverse_iterator
    mut_rbegin()
      { return reverse_iterator(this->mut_end());  }

    reverse_iterator
    mut_rend()
      { return reverse_iterator(this->mut_begin());  }

    // modifiers
    Option_Map&
    clear() noexcept
      {
        this->m_stor.clear();
        this->m_nbkt = 0;
        return *this;
      }

    Option_Map&
    swap(Option_Map& other) noexcept
      {
        this->m_stor.swap(other.m_stor);
        ::std::swap(this->m_nbkt, other.m_nbkt);
        return *this;
      }

    // Get all values with a given key.
    // The return value is a pair of pointers denoting the beginning and end of
    // the value array. A scalar value is considered to be an array of only one
    // value.
    pair<const cow_string*, const cow_string*>
    range(cow_string::shallow_type key) const noexcept
      { return this->do_range_hint(key, details_option_map::ci_hash()(key));  }

    pair<const cow_string*, const cow_string*>
    range(const cow_string& key) const noexcept
      { return this->range(sref(key));  }

    template<typename FuncT>
    void
    for_each(cow_string::shallow_type key, FuncT&& func) const
      {
        auto r = this->range(key);
        while(r.first != r.second)
          ::std::forward<FuncT>(func)(*(r.first++));
      }

    template<typename FuncT>
    void
    for_each(const cow_string& key, FuncT&& func) const
      { return this->for_each(sref(key), ::std::forward<FuncT>(func));  }

    pair<cow_string*, cow_string*>
    mut_range(cow_string::shallow_type key)
      { return this->do_mut_range_hint(key, details_option_map::ci_hash()(key));  }

    pair<cow_string*, cow_string*>
    mut_range(const cow_string& key)
      { return this->mut_range(sref(key));  }

    // Erase all values with a given key.
    // The return value is the number of values that have been erased.
    // N.B. These functions might throw `std::bad_alloc`.
    size_t
    erase(cow_string::shallow_type key)
      { return this->do_erase_hint(key, details_option_map::ci_hash()(key));  }

    size_t
    erase(const cow_string& key)
      { return this->erase(sref(key));  }

    // Get a scalar value.
    // If multiple values exist, the last one is returned.
    const cow_string*
    find_opt(cow_string::shallow_type key) const noexcept
      { return details_option_map::range_back(this->range(key));  }

    const cow_string*
    find_opt(const cow_string& key) const noexcept
      { return this->find_opt(sref(key));  }

    template<typename PredT>
    const cow_string*
    find_if_opt(cow_string::shallow_type key, PredT&& pred) const
      {
        auto r = this->range(key);
        while(r.first != r.second)
          if(::std::forward<PredT>(pred)(*--(r.second)))
            return r.second;
        return nullptr;
      }

    template<typename PredT>
    const cow_string*
    find_if_opt(const cow_string& key, PredT&& pred) const
      { return this->find_if_opt(sref(key), ::std::forward<PredT>(pred));  }

    cow_string*
    mut_find_opt(cow_string::shallow_type key)
      { return details_option_map::range_back(this->mut_range(key));  }

    cow_string*
    mut_find_opt(const cow_string& key)
      { return this->mut_find_opt(sref(key));  }

    // Set a scalar value.
    cow_string&
    open(const cow_string& key)
      { return this->do_open_hint(key, details_option_map::ci_hash()(key));  }

    template<typename StringT>
    cow_string&
    set(const cow_string& key, StringT&& str)
      { return this->open(key) = ::std::forward<StringT>(str);  }

    // Get the number of values with the given key.
    size_t
    count(cow_string::shallow_type key) const noexcept
      { return details_option_map::range_size(this->range(key));  }

    size_t
    count(const cow_string& key) const noexcept
      { return this->count(sref(key));  }

    // Append a value to an array.
    cow_string&
    append(const cow_string& key)
      { return this->do_append_hint(key, details_option_map::ci_hash()(key));  }

    template<typename StringT>
    cow_string&
    append(const cow_string& key, StringT&& str)
      { return this->append(key) = ::std::forward<StringT>(str);  }

    // Converts this map to a human-readable string.
    tinyfmt&
    print(tinyfmt& fmt) const;

    // Converts this map to an URL query string.
    // This is the inverse function of `parse_url_query()`.
    tinyfmt&
    print_url_query(tinyfmt& fmt) const;

    // Parses a URL query string.
    // If an element contains no equals sign (such as the `foo` in `foo&bar=42`),
    // it is interpreted as a key having an empty value.
    // An exception is thrown if the string is invalid.
    Option_Map&
    parse_url_query(const cow_string& str);

    // Converts this map to an HTTP header value.
    // All values with empty keys are output first; the others follow.
    // All non-empty keys must be valid tokens according to RFC 7230.
    // This is the inverse function of `parse_http_header()`.
    tinyfmt&
    print_http_header(tinyfmt& fmt) const;

    // Parses an HTTP header value.
    // A header value is a series of fields delimited by semicolons. A field may
    // be a plain string (non-option) or an option comprising a key followed by
    // an optional equals sign and its value. Non-options are parsed as values
    // with empty keys. `nonopts` specifies the maximum number of non-options to
    // accept. All non-empty keys must be valid tokens according to RFC 7230. If
    // `comma_opt` is null, commas are parsed as part of values. This mode allows
    // parsing values of `Set-Cookie:` headers. If `comma_opt` is non-null,
    // parsing starts from `*comma_opt` and a header value is terminated by a
    // non-quoted comma, whose offset is then stored into `comma_opt`. This means
    // the caller should initialize the value of `*comma_opt` to zero, then call
    // this function to parse and process header values repeatedly, until
    // `*comma_opt` reaches `str.size()`.
    // An exception is thrown if the string is invalid.
    Option_Map&
    parse_http_header(size_t* comma_opt, const cow_string& str, size_t nonopts);
  };

inline void
swap(Option_Map& lhs, Option_Map& rhs) noexcept
  { lhs.swap(rhs);  }

inline tinyfmt&
operator<<(tinyfmt& fmt, const Option_Map& map)
  { return map.print(fmt);  }

}  // namespace poseidon

#endif
