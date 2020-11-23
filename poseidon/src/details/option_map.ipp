// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_HTTP_OPTION_MAP_HPP_
#  error Please include <poseidon/http/option_map.hpp> instead.
#endif

namespace poseidon {
namespace details_option_map {

struct ci_hash
  {
    template<typename StringT>
    constexpr
    size_t
    operator()(const StringT& str)
      const
      noexcept(noexcept(::std::declval<const StringT&>().c_str()) &&
               noexcept(::std::declval<const StringT&>().length()))
      {
        return ::rocket::ascii_ci_hash(str.c_str(), str.length());
      }
  };

struct ci_equal
  {
    template<typename StringT, typename OtherT>
    constexpr
    bool
    operator()(const StringT& str, const OtherT& oth)
      const
      noexcept(noexcept(::std::declval<const StringT&>().c_str()) &&
               noexcept(::std::declval<const StringT&>().length()) &&
               noexcept(::std::declval<const OtherT&>().c_str()) &&
               noexcept(::std::declval<const OtherT&>().length()))
      {
        return ::rocket::ascii_ci_equal(str.c_str(), str.length(),
                                        oth.c_str(), oth.length());
      }
  };

class Bucket
  {
  private:
    ::rocket::basic_prehashed_string<
                   cow_string, ci_hash, ci_equal> m_key;

    ::rocket::variant<nullopt_t, cow_string,
                ::rocket::cow_vector<cow_string>> m_val;

  public:
    constexpr
    Bucket()
      noexcept
      = default;

  public:
    constexpr
    const cow_string&
    key()
      const noexcept
      { return this->m_key.rdstr();  }

    const cow_string&
    set_key(const cow_string& str)
      noexcept
      { return this->m_key.assign(str).rdstr();  }

    template<typename OtherT>
    constexpr
    bool
    key_equals(const OtherT& oth)
      const
      { return this->m_key == oth;  }

    constexpr
    size_t
    hash()
      const noexcept
      { return this->m_key.rdhash();  }

    constexpr operator
    bool()
      const noexcept
      { return this->m_val.index() != 0;  }

    Bucket&
    reset()
      noexcept
      {
        this->m_key.clear();
        this->m_val = nullopt;
        return *this;
      }

    ROCKET_PURE_FUNCTION
    pair<const cow_string*, const cow_string*>
    range()
      const noexcept
      {
        if(this->m_val.index() == 0)  // nothing
          return { nullptr, nullptr };

        if(this->m_val.index() == 1) {  // scalar
          const auto& str = this->m_val.as<1>();
          return { &str, &str + 1 };
        }

        // array
        ROCKET_ASSERT(this->m_val.index() == 2);
        const auto& arr = this->m_val.as<2>();
        auto ptr = arr.data();
        return { ptr, ptr + arr.size() };
      }

    pair<cow_string*, cow_string*>
    mut_range()
      {
        if(this->m_val.index() == 0)  // nothing
          return { nullptr, nullptr };

        if(this->m_val.index() == 1) {  // scalar
          auto& str = this->m_val.as<1>();
          return { &str, &str + 1 };
        }

        // array
        ROCKET_ASSERT(this->m_val.index() == 2);
        auto& arr = this->m_val.as<2>();
        auto ptr = arr.mut_data();
        return { ptr, ptr + arr.size() };
      }

    ROCKET_PURE_FUNCTION
    size_t
    count()
      const noexcept
      {
        if(this->m_val.index() == 0)  // nothing
          return 0;

        if(this->m_val.index() == 1)  // scalar
          return 1;

        // array
        ROCKET_ASSERT(this->m_val.index() == 2);
        return this->m_val.as<2>().size();
      }

    cow_string&
    mut_scalar()
      {
        if(ROCKET_EXPECT(this->m_val.index() == 0))  // nothing
          return this->m_val.emplace<1>();

        if(ROCKET_EXPECT(this->m_val.index() == 1))  // scalar
          return this->m_val.as<1>();

        // array
        ROCKET_ASSERT(this->m_val.index() == 2);
        auto& arr = this->m_val.as<2>();
        if(arr.empty())
          return this->m_val.emplace<1>();

        cow_string str = ::std::move(arr.mut_back());
        return this->m_val.emplace<1>(::std::move(str));
      }

    ::rocket::cow_vector<cow_string>&
    mut_array()
      {
        if(ROCKET_EXPECT(this->m_val.index() == 0))  // nothing
          return this->m_val.emplace<2>();

        if(ROCKET_EXPECT(this->m_val.index() == 2))  // array
          return this->m_val.as<2>();

        // scalar
        ROCKET_ASSERT(this->m_val.index() == 1);
        auto& str = this->m_val.as<1>();

        ::rocket::cow_vector<cow_string> arr;
        arr.emplace_back(::std::move(str));
        return this->m_val.emplace<2>(::std::move(arr));
      }

    // These two functions are used by iterators.
    pair<const cow_string*, const cow_string*>
    xlocal_range()
      const noexcept
      { return this->range();  }

    pair<cow_string*, cow_string*>
    xlocal_range()
      { return this->mut_range();  }
  };

template<typename valueT,
         typename bucketT = typename ::rocket::copy_cv<Bucket, valueT>::type>
class Iterator
  {
    friend Option_Map;

  public:
    using iterator_category  = ::std::bidirectional_iterator_tag;
    using value_type         = pair<cow_string, cow_string>;
    using pointer            = const pair<const cow_string&, valueT&>*;
    using reference          = pair<const cow_string&, valueT&>;
    using difference_type    = ptrdiff_t;

  private:
    bucketT* m_begin;
    bucketT* m_cur;
    bucketT* m_end;

    valueT* m_local_begin;
    pair<const cow_string*, valueT*> m_local_ptrs;
    valueT* m_local_end;

  private:
    // This constructor is called by the container.
    Iterator(bucketT* begin, size_t ncur, size_t nend)
      noexcept
      : m_begin(begin), m_cur(begin + ncur), m_end(begin + nend)
      {
        // Go to the first following non-empty bucket if any.
        this->do_clear_local_ptrs();
        while((this->m_cur != this->m_end) && !this->do_set_local_ptrs())
          this->m_cur++;
      }

    void
    do_clear_local_ptrs()
      noexcept
      {
        this->m_local_begin = nullptr;
        this->m_local_ptrs.first = nullptr;
        this->m_local_ptrs.second = nullptr;
        this->m_local_end = nullptr;
      }

    bool
    do_set_local_ptrs()
      noexcept
      {
        auto r = this->m_cur->xlocal_range();
        if(r.first == r.second)
          return false;

        this->m_local_begin = r.first;
        this->m_local_ptrs.first = &(this->m_cur->key());
        this->m_local_ptrs.second = r.first;
        this->m_local_end = r.second;
        return true;
      }

  public:
    constexpr
    Iterator()
      noexcept
      : m_begin(), m_cur(), m_end(),
        m_local_begin(), m_local_ptrs(), m_local_end()
      { }

    template<typename yvalueT, typename ybucketT,
    ROCKET_ENABLE_IF(::std::is_convertible<ybucketT*, bucketT*>::value)>
    constexpr
    Iterator(const Iterator<yvalueT, ybucketT>& other)
      noexcept
      : m_begin(other.m_begin),
        m_cur(other.m_cur),
        m_end(other.m_end),
        m_local_begin(other.m_local_begin),
        m_local_ptrs(other.m_local_ptrs),
        m_local_end(other.m_local_end)
      { }

    template<typename yvalueT, typename ybucketT,
    ROCKET_ENABLE_IF(::std::is_convertible<ybucketT*, bucketT*>::value)>
    Iterator&
    operator=(const Iterator<yvalueT, ybucketT>& other)
      noexcept
      {
        this->m_begin = other.m_begin;
        this->m_cur = other.m_cur;
        this->m_end = other.m_end;
        this->m_local_begin = other.m_local_begin;
        this->m_local_ptrs = other.m_local_ptrs;
        this->m_local_end = other.m_local_end;
        return *this;
      }

  private:
    const pair<const cow_string&, valueT&>*
    do_validate(bool deref)
      const noexcept
      {
        ROCKET_ASSERT_MSG(this->m_begin, "Iterator not initialized");
        ROCKET_ASSERT_MSG((this->m_begin <= this->m_cur) && (this->m_cur <= this->m_end),
                          "Iterator out of range");
        ROCKET_ASSERT_MSG(!deref || (this->m_cur < this->m_end),
                          "Past-the-end iterator not dereferenceable");
        ROCKET_ASSERT_MSG(!deref || ((this->m_cur->range().first == this->m_local_begin) &&
                                     (this->m_cur->range().second == this->m_local_end) &&
                                     (this->m_local_begin <= this->m_local_ptrs.second) &&
                                     (this->m_local_ptrs.second <= this->m_local_end)),
                          "Iterator invalidated");

        // XXX: Is it really necessary to support `operator->`?
        return reinterpret_cast<const pair<decltype(*(this->m_local_ptrs.first)),
                     decltype(*(this->m_local_ptrs.second))>*>(&(this->m_local_ptrs));
      }

    Iterator
    do_next()
      const noexcept
      {
        ROCKET_ASSERT_MSG(this->m_begin, "Iterator not initialized");

        auto res = *this;
        if(ROCKET_EXPECT((res.m_local_ptrs.second < this->m_local_end) &&
                         (++(res.m_local_ptrs.second) != this->m_local_end)))
          return res;

        res.do_clear_local_ptrs();
        do {
          ROCKET_ASSERT_MSG(res.m_cur != this->m_end,
                            "Past-the-end iterator not incrementable");
          res.m_cur++;
        }
        while((res.m_cur != this->m_end) && !res.do_set_local_ptrs());
        return res;
      }

    Iterator
    do_prev()
      const noexcept
      {
        ROCKET_ASSERT_MSG(this->m_begin, "Iterator not initialized");

        auto res = *this;
        if(ROCKET_EXPECT((res.m_local_ptrs.second > this->m_local_begin) &&
                         (--(res.m_local_ptrs.second) != this->m_local_begin)))
          return res;

        res.do_clear_local_ptrs();
        do {
          ROCKET_ASSERT_MSG(res.m_cur != this->m_begin,
                            "Beginning iterator not decrementable");
          res.m_cur--;
        }
        while(!res.do_set_local_ptrs());
        return res;
      }

  public:
    reference
    operator*()
      const noexcept
      { return *(this->do_validate(true));  }

    pointer
    operator->()
      const noexcept
      { return this->do_validate(true);  }

    Iterator&
    operator++()
      noexcept
      { return *this = this->do_next();  }

    Iterator&
    operator--()
      noexcept
      { return *this = this->do_prev();  }

    Iterator
    operator++(int)
      noexcept
      { return ::std::exchange(*this, this->do_next());  }

    Iterator
    operator--(int)
      noexcept
      { return ::std::exchange(*this, this->do_prev());  }

    template<typename ybucketT>
    constexpr
    bool
    operator==(const Iterator<ybucketT>& other)
      const noexcept
      { return this->m_local_ptrs.second == other.m_local_ptrs.second;  }

    template<typename ybucketT>
    constexpr
    bool
    operator!=(const Iterator<ybucketT>& other)
      const noexcept
      { return this->m_local_ptrs.second != other.m_local_ptrs.second;  }
  };

}  // namespace details_option_map
}  // namespace poseidon
