// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#include "precompiled.hpp"
#include "utils.hpp"

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

bool
ascii_ci_has_token(const cow_string& str, char delim, const char* tok, size_t len)
  {
    size_t epos = 0;
    while(epos < str.size()) {
      // Get a token.
      size_t bpos = epos;
      epos = ::std::min(str.find(bpos, delim), str.size()) + 1;
      size_t mpos = epos - 1;

      // Skip leading and trailing blank characters, if any.
      while((bpos != mpos) && ::rocket::is_any_of(str[bpos], {' ', '\t'}))
        bpos++;

      while((bpos != mpos) && ::rocket::is_any_of(str[mpos-1], {' ', '\t'}))
        mpos--;

      // If the token matches `close`, the connection shall be closed.
      if(::rocket::ascii_ci_equal(str.data() + bpos, mpos - bpos, tok, len))
        return true;
    }
    return false;
  }

size_t
ascii_explode(::rocket::cow_vector<cow_string>& segments, const cow_string& text,
              char delim, size_t limit)
  {
    segments.clear();
    size_t epos = 0;
    while(epos < text.size()) {
      // Get a token.
      size_t bpos = epos;
      epos = (segments.size() + 1 >= limit) ? text.size()
                 : ::std::min(text.find(bpos, delim), text.size()) + 1;
      size_t mpos = epos - 1;

      // Skip leading and trailing blank characters, if any.
      while((bpos != mpos) && ::rocket::is_any_of(text[bpos], {' ', '\t'}))
        bpos++;

      while((bpos != mpos) && ::rocket::is_any_of(text[mpos-1], {' ', '\t'}))
        mpos--;

      // Push this token.
      segments.emplace_back(text.data() + bpos, mpos - bpos);
    }
    return segments.size();
  }

size_t
ascii_implode(cow_string& text, const ::rocket::cow_vector<cow_string>& segments,
              char delim)
  {
    text.clear();
    if(segments.size()) {
      // Write the first token.
      text << segments[0];

      // Write the other tokens, each of which is preceded by a delimiter.
      for(size_t k = 1;  k < segments.size();  ++k)
        text << delim << segments[k];
    }
    return segments.size();
  }

}  // namespace poseidon
