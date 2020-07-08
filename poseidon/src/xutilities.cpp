// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#include "precompiled.hpp"
#include "xutilities.hpp"

namespace poseidon {

cow_string
ascii_uppercase(cow_string str)
  {
    // Only modify the string when it really has to modified.
    for(size_t k = 0;  k != str.size();  ++k) {
      char ch = str[k];
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
      char ch = str[k];
      if(('A' <= ch) && (ch <= 'Z'))
        str.mut(k) = static_cast<char>(ch + 0x20);
    }
    return ::std::move(str);
  }

cow_string
ascii_trim(cow_string str)
  {
    // Return an empty string if it comprises only blank characters.
    size_t bp = str.find_first_not_of(" \t");
    if(bp == cow_string::npos)
      return { };

    // Get the offset of the past-the-last character.
    size_t ep = str.find_last_not_of(" \t") + 1;

    // Only modify the string when it really has to modified.
    if((bp != 0) || (ep != str.size())) {
      str.erase(0, bp);
      str.erase(ep - bp);
    }
    return ::std::move(str);
  }

}  // namespace poseidon
