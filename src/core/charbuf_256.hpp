// This file is part of Poseidon.
// Copyleft 2022, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_CORE_CHARBUF_256_
#define POSEIDON_CORE_CHARBUF_256_

#include "../fwd.hpp"

namespace poseidon {

// This class provides 256-byte storage for temporary use.
class charbuf_256
  {
  private:
    char m_data[256];

  public:
    // Initializes a null-terminated string of zero characters.
    explicit
    charbuf_256() noexcept
      {
        this->m_data[0] = 0;
      }

   public:
     // Swaps two buffers.
     charbuf_256&
     swap(charbuf_256& other) noexcept
       {
         char temp[256];
         ::std::memcpy(temp, other.m_data, sizeof(temp));
         ::std::memcpy(other.m_data, this->m_data, sizeof(temp));
         ::std::memcpy(this->m_data, temp, sizeof(temp));
         return *this;
       }

     // Performs 3-way comparison of two buffers.
     int
     compare(const charbuf_256& other) const noexcept
       {
         return ::std::strcmp(this->m_data, other.m_data);
       }

     int
     compare(const char* other) const noexcept
       {
         return ::std::strcmp(this->m_data, other);
       }

     // Returns a pointer to internal storage so a buffer can be passed as
     // an argument for `char*`.
     constexpr operator
     const char*() const noexcept
       { return this->m_data;  }

     operator
     char*() noexcept
       { return this->m_data;  }
   };

inline
void
swap(charbuf_256& lhs, charbuf_256& rhs) noexcept
  {
    lhs.swap(rhs);
  }

inline
bool
operator==(const charbuf_256& lhs, const charbuf_256& rhs) noexcept
  {
    return ::std::strcmp(lhs, rhs) == 0;
  }

inline
bool
operator==(const char* lhs, const charbuf_256& rhs) noexcept
  {
    return ::std::strcmp(lhs, rhs) == 0;
  }

inline
bool
operator==(const charbuf_256& lhs, const char* rhs) noexcept
  {
    return ::std::strcmp(lhs, rhs) == 0;
  }

inline
bool
operator!=(const charbuf_256& lhs, const charbuf_256& rhs) noexcept
  {
    return ::std::strcmp(lhs, rhs) != 0;
  }

inline
bool
operator!=(const char* lhs, const charbuf_256& rhs) noexcept
  {
    return ::std::strcmp(lhs, rhs) != 0;
  }

inline
bool
operator!=(const charbuf_256& lhs, const char* rhs) noexcept
  {
    return ::std::strcmp(lhs, rhs) != 0;
  }

inline
bool
operator<(const charbuf_256& lhs, const charbuf_256& rhs) noexcept
  {
    return ::std::strcmp(lhs, rhs) < 0;
  }

inline
bool
operator<(const char* lhs, const charbuf_256& rhs) noexcept
  {
    return ::std::strcmp(lhs, rhs) < 0;
  }

inline
bool
operator<(const charbuf_256& lhs, const char* rhs) noexcept
  {
    return ::std::strcmp(lhs, rhs) < 0;
  }

inline
bool
operator>(const charbuf_256& lhs, const charbuf_256& rhs) noexcept
  {
    return ::std::strcmp(lhs, rhs) > 0;
  }

inline
bool
operator>(const char* lhs, const charbuf_256& rhs) noexcept
  {
    return ::std::strcmp(lhs, rhs) > 0;
  }

inline
bool
operator>(const charbuf_256& lhs, const char* rhs) noexcept
  {
    return ::std::strcmp(lhs, rhs) > 0;
  }

inline
bool
operator<=(const charbuf_256& lhs, const charbuf_256& rhs) noexcept
  {
    return ::std::strcmp(lhs, rhs) <= 0;
  }

inline
bool
operator<=(const char* lhs, const charbuf_256& rhs) noexcept
  {
    return ::std::strcmp(lhs, rhs) <= 0;
  }

inline
bool
operator<=(const charbuf_256& lhs, const char* rhs) noexcept
  {
    return ::std::strcmp(lhs, rhs) <= 0;
  }

inline
bool
operator>=(const charbuf_256& lhs, const charbuf_256& rhs) noexcept
  {
    return ::std::strcmp(lhs, rhs) >= 0;
  }

inline
bool
operator>=(const char* lhs, const charbuf_256& rhs) noexcept
  {
    return ::std::strcmp(lhs, rhs) >= 0;
  }

inline
bool
operator>=(const charbuf_256& lhs, const char* rhs) noexcept
  {
    return ::std::strcmp(lhs, rhs) >= 0;
  }

}  // namespace poseidon

#endif
