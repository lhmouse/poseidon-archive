// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#include "precompiled.hpp"
#include "util.hpp"

namespace poseidon {

cow_string
ascii_uppercase(cow_string str)
  {
    // Only modify the string when it really has to modified.
    for(size_t k = 0;  k != str.size();  ++k) {
      char32_t ch = static_cast<uint8_t>(str[k]);
      if(('a' <= ch) && (ch <= 'z'))
        str.mut(k) = static_cast<char>(ch - 0x20);
    }
    return ::std::move(str);
  }

cow_string
ascii_lowercase(cow_string str)
  {
    // Only modify the string when it really has to modified.
    for(size_t k = 0;  k != str.size();  ++k) {
      char32_t ch = static_cast<uint8_t>(str[k]);
      if(('A' <= ch) && (ch <= 'Z'))
        str.mut(k) = static_cast<char>(ch + 0x20);
    }
    return ::std::move(str);
  }

cow_string
ascii_trim(cow_string str)
  {
    // Remove leading blank characters.
    // Return an empty string if all characters are blank.
    size_t k = cow_string::npos;
    for(;;) {
      if(++k == str.size())
        return { };

      char32_t ch = static_cast<uint8_t>(str[k]);
      if((ch != ' ') && (ch != '\t'))
        break;
    }
    if(k != 0)
      str.erase(0, k);

    // Remove trailing blank characters.
    k = str.size();
    for(;;) {
      if(--k == 0)
        break;

      char32_t ch = static_cast<uint8_t>(str[k]);
      if((ch != ' ') && (ch != '\t'))
        break;
    }
    if(++k != str.size())
      str.erase(k);

    return ::std::move(str);
  }

}  // namespace poseidon
