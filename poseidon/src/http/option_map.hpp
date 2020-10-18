// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_HTTP_OPTION_MAP_HPP_
#define POSEIDON_HTTP_OPTION_MAP_HPP_

#include "../fwd.hpp"

namespace poseidon {

class Option_Map
  {
  private:
    struct Bucket
      {
        // Keys are case-insensitive.
        cow_string key;
        size_t hash;

        // Values are strings or array of strings.
        ::rocket::variant<nullopt_t, cow_string,
              ::rocket::cow_vector<cow_string>> vstor;

        operator
        bool()
          const noexcept
          { return this->vstor.index() != 0;  }
      };

    ::rocket::cow_vector<Bucket> m_stor;
    size_t m_nbkt = 0;

  public:
    constexpr
    Option_Map()
      noexcept
      { }

    ASTERIA_COPYABLE_DESTRUCTOR(Option_Map);

  private:
    // This function may be inlined in case of string literals.
    static constexpr
    size_t
    do_key_hash(cow_string::shallow_type sh)
      noexcept
      {
        auto bptr = sh.c_str();
        auto eptr = bptr + sh.length();

        // Implement the FNV-1a hash algorithm.
        char32_t reg = 0x811C9DC5;
        while(bptr != eptr) {
          char32_t ch = static_cast<uint8_t>(*(bptr++));

          // Upper-case letters are converted to corresponding lower-case ones,
          // so this algorithm is case-insensitive.
          ch |= 0x20;
          reg = (reg ^ ch) * 0x1000193;
        }
        return reg;
      }

    static constexpr
    bool
    do_key_equal(cow_string::shallow_type s1, cow_string::shallow_type s2)
      noexcept
      {
        if(s1.length() != s2.length())
          return false;

        auto bptr = s1.c_str();
        auto eptr = bptr + s1.length();
        auto kptr = s2.c_str();
        if(bptr == kptr)
          return true;

        // Perform character-wise comparison.
        while(bptr != eptr) {
          char32_t ch = static_cast<uint8_t>(*(bptr++));
          char32_t cmp = ch ^ static_cast<uint8_t>(*(kptr++));
          if(cmp == 0)
            continue;

          // Upper-case letters are converted to corresponding lower-case ones,
          // so this algorithm is case-insensitive.
          ch |= 0x20;
          if((cmp != 0x20) || (ch < 'a') || ('z' < ch))
            return false;
        }
        return true;
      }

    ROCKET_PURE_FUNCTION
    inline
    size_t
    do_bucket_index(cow_string::shallow_type key, size_t hash)
      const noexcept;

    ROCKET_PURE_FUNCTION
    pair<const cow_string*, size_t>
    do_equal_range(cow_string::shallow_type key, size_t hash)
      const noexcept;

    inline
    void
    do_reserve_more();

    cow_string&
    do_mutable_scalar(const cow_string& key, size_t hash);

    cow_string&
    do_mutable_append(const cow_string& key, size_t hash);

    size_t
    do_erase(cow_string::shallow_type key, size_t hash);

  public:
    // Checks whether any buckets are in use.
    bool
    empty()
      const noexcept
      { return this->m_nbkt != 0;  }

    // Gets a scalar value.
    // If multiple values exist, the last one is returned.
    const cow_string*
    find_opt(cow_string::shallow_type key)
      const noexcept
      {
        auto r = this->do_equal_range(key, this->do_key_hash(key));
        return r.second ? (r.first + r.second - 1) : nullptr;
      }

    const cow_string*
    find_opt(const cow_string& key)
      const noexcept
      { return this->find_opt(::rocket::sref(key));  }

    // Sets a scalar value.
    // Existent values are erased.
    template<typename... ArgsT>
    cow_string&
    set(const cow_string& key, ArgsT&&... args)
      {
        auto& s = this->do_mutable_scalar(key, this->do_key_hash(::rocket::sref(key)));
        return s.assign(::std::forward<ArgsT>(args)...);
      }

    // Gets an array value.
    pair<const cow_string*, const cow_string*>
    equal_range(cow_string::shallow_type key)
      const noexcept
      {
        auto r = this->do_equal_range(key, this->do_key_hash(key));
        return { r.first, r.first + r.second };
      }

    pair<const cow_string*, const cow_string*>
    equal_range(const cow_string& key)
      const noexcept
      { return this->equal_range(::rocket::sref(key));  }

    // Appends a new value to an array.
    // If a scalar value exists, it is converted to an array.
    template<typename... ArgsT>
    cow_string&
    push(const cow_string& key, ArgsT&&... args)
      {
        auto& s = this->do_mutable_append(key, this->do_key_hash(::rocket::sref(key)));
        return s.assign(::std::forward<ArgsT>(args)...);
      }

    // Retrieves the number of values matching a key.
    size_t
    count(cow_string::shallow_type key)
      const noexcept
      {
        auto r = this->do_equal_range(key, this->do_key_hash(key));
        return r.second;
      }

    size_t
    count(const cow_string& key)
      const noexcept
      { return this->count(::rocket::sref(key));  }

    // Removes all values matching a key.
    // Returns the number of buckets that have been removed.
    size_t
    erase(cow_string::shallow_type key)
      {
        auto n = this->do_erase(key, this->do_key_hash(key));
        return n;
      }

    size_t
    erase(const cow_string& key)
      { return this->erase(::rocket::sref(key));  }

    // Invokes the given function with all key-value pairs.
    // If the function returns `false`, this function stops to return `false`.
    // Otherwise, this function returns `true`.
    template<typename FuncT>
    bool
    enumerate(FuncT&& func)
      const
      {
        for(const auto& bkt : this->m_stor) {
          if(bkt.vstor.index() == 1) {
            // Process a scalar value.
            if(!func(bkt.key, bkt.vstor.as<1>()))
              return false;
          }
          else if(bkt.vstor.index() == 2) {
            // Process every value in the array.
            for(const auto& s : bkt.vstor.as<2>())
              if(!func(bkt.key, s))
                return false;
          }
        }
        return true;
      }

    // These are general modifiers.
    Option_Map&
    clear()
      noexcept
      {
        this->m_stor.clear();
        this->m_nbkt = 0;
        return *this;
      }

    Option_Map&
    swap(Option_Map& other)
      noexcept
      {
        this->m_stor.swap(other.m_stor);
        ::std::swap(this->m_nbkt, other.m_nbkt);
        return *this;
      }

    // Converts this map to a human-readable string.
    tinyfmt&
    print(tinyfmt& fmt)
      const;

    // Converts this map to an URL query string.
    // This is the inverse function of `parse_url_query()`.
    tinyfmt&
    print_url_query(tinyfmt& fmt)
      const;

    // Parses a URL query string.
    // If an element contains no equals sign (such as the `foo` in `foo&bar=42`), it is
    // interpreted as a key having an empty value.
    // An exception is thrown if the string is invalid.
    Option_Map&
    parse_url_query(const cow_string& str);

    // Converts this map to an HTTP header value.
    // All values with empty keys are output first; the others follow.
    // All non-empty keys must be valid tokens according to RFC 7230.
    // This is the inverse function of `parse_http_header()`.
    tinyfmt&
    print_http_header(tinyfmt& fmt)
      const;

    // Parses an HTTP header value.
    // A header value is a series of fields delimited by semicolons. A field may be a
    // plain string (non-option) or an option comprising a key followed by an optional
    // equals sign and its value. Non-options are parsed as values with empty keys.
    // `nonopts` specifies the maximum number of non-options to accept.
    // All non-empty keys must be valid tokens according to RFC 7230.
    // If `comma_opt` is null, commas are parsed as part of values. This mode allows
    // parsing values of `Set-Cookie:` headers.
    // If `comma_opt` is non-null, parsing starts from `*comma_opt` and a header value
    // is terminated by a non-quoted comma, whose offset is then stored into `comma_opt`.
    // This means the caller should initialize the value of `*comma_opt` to zero, then
    // call this function to parse and process header values repeatedly, until
    // `*comma_opt` reaches `str.size()`.
    // An exception is thrown if the string is invalid.
    Option_Map&
    parse_http_header(size_t* comma_opt, const cow_string& str, size_t nonopts);
  };

inline
void
swap(Option_Map& lhs, Option_Map& rhs)
  noexcept
  { lhs.swap(rhs);  }

inline
tinyfmt&
operator<<(tinyfmt& fmt, const Option_Map& map)
  { return map.print(fmt);  }

}  // namespace poseidon

#endif
