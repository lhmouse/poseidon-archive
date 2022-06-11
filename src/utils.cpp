// This file is part of Poseidon.
// Copyleft 2022, LH_Mouse. All wrongs reserved.

#include "precompiled.ipp"
#include "utils.hpp"
#include <execinfo.h>  // backtrace();

namespace poseidon {

void
throw_runtime_error_with_backtrace(const char* file, long line, const char* func, cow_string&& msg)
  {
    // Compose the string to throw.
    cow_string data;
    data.reserve(2047);

    // Append the function name.
    data += func;
    data += ": ";

    // Append the user-provided exception message.
    data.append(msg.begin(), msg.end());

    // Remove trailing space characters.
    size_t pos = data.find_last_not_of(" \f\n\r\t\v");
    data.erase(pos + 1);
    data += "\n";

    // Append the source location.
    data += "[thrown from '";
    data += file;
    data += ':';
    ::rocket::ascii_numput nump;
    nump.put_DU((unsigned long) line);
    data.append(nump.data(), nump.size());
    data += "']";

    // Backtrace frames.
    ::rocket::unique_ptr<char*, void (void*)> bt_syms(::free);

    array<void*, 32> bt_frames;
    int nframes = ::backtrace(bt_frames.data(), (int) bt_frames.size());
    if(nframes > 0)
      bt_syms.reset(::backtrace_symbols(bt_frames.data(), nframes));

    if(bt_syms) {
      // Determine the width of the frame index field.
      nump.put_DU((uint32_t) nframes - 1);
      static_vector<char, 24> sbuf(nump.size(), ' ');
      sbuf.emplace_back();

      // Append stack frames.
      data += "\n[backtrace frames:\n  ";
      for(size_t k = 0;  k != (uint32_t) nframes;  ++k) {
        nump.put(k);
        ::std::reverse_copy(nump.begin(), nump.end(), sbuf.mut_rbegin() + 1);
        data += sbuf.data();
        data += ") ";
        data += bt_syms[k];
        data += "\n  ";
      }
      data += "-- end of backtrace frames]";
    }
    else
      data += "\n[no backtrace available]";

    // Throw it.
    throw ::std::runtime_error(data.c_str());
  }

cow_string
ascii_uppercase(cow_string text)
  {
    // Only modify the string when it really has to modified.
    for(size_t k = 0;  k != text.size();  ++k) {
      char32_t ch = (uint8_t) text[k];
      if(('a' <= ch) && (ch <= 'z'))
        text.mut(k) = (char) (ch - 0x20);
    }
    return ::std::move(text);
  }

cow_string
ascii_lowercase(cow_string text)
  {
    // Only modify the string when it really has to modified.
    for(size_t k = 0;  k != text.size();  ++k) {
      char32_t ch = (uint8_t) text[k];
      if(('A' <= ch) && (ch <= 'Z'))
        text.mut(k) = (char) (ch + 0x20);
    }
    return ::std::move(text);
  }

cow_string
ascii_trim(cow_string text)
  {
    // Remove leading blank characters.
    // Return an empty string if all characters are blank.
    size_t k = cow_string::npos;
    for(;;) {
      if(++k == text.size())
        return { };

      char32_t ch = (uint8_t) text[k];
      if((ch != ' ') && (ch != '\t'))
        break;
    }
    if(k != 0)
      text.erase(0, k);

    // Remove trailing blank characters.
    k = text.size();
    for(;;) {
      if(--k == 0)
        break;

      char32_t ch = (uint8_t) text[k];
      if((ch != ' ') && (ch != '\t'))
        break;
    }
    if(++k != text.size())
      text.erase(k);

    return ::std::move(text);
  }

bool
ascii_ci_has_token(const cow_string& text, char delim, const char* token, size_t len)
  {
    size_t bpos = text.find_first_not_of(" \t");
    while(bpos < text.size()) {
      // Get the end of this segment.
      // If the delimiter is not found, make sure `epos` is reasonably large
      // and incrementing it will not overflow.
      size_t epos = text.find(bpos, delim) * 2 / 2;

      // Skip trailing blank characters, if any.
      size_t mpos = text.find_last_not_of(epos - 1, " \t");
      ROCKET_ASSERT(mpos != text.npos);
      if(::rocket::ascii_ci_equal(text.data() + bpos, mpos + 1 - bpos, token, len))
        return true;

      // Skip the delimiter and blank characters that follow it.
      bpos = text.find_first_not_of(epos + 1, " \t");
    }
    return false;
  }

size_t
explode(cow_vstrings& segments, const cow_string& text, char delim, size_t limit)
  {
    segments.clear();
    size_t bpos = text.find_first_not_of(" \t");
    while(bpos < text.size()) {
      // Get the end of this segment.
      // If the delimiter is not found, make sure `epos` is reasonably large
      // and incrementing it will not overflow.
      size_t epos = text.npos / 2;
      if(segments.size() + 1 < limit)
        epos = text.find(bpos, delim) * 2 / 2;

      // Skip trailing blank characters, if any.
      size_t mpos = text.find_last_not_of(epos - 1, " \t");
      ROCKET_ASSERT(mpos != text.npos);
      segments.emplace_back(text.data() + bpos, mpos + 1 - bpos);

      // Skip the delimiter and blank characters that follow it.
      bpos = text.find_first_not_of(epos + 1, " \t");
    }
    return segments.size();
  }

size_t
implode(cow_string& text, const cow_vstrings& segments, char delim)
  {
    text.clear();
    if(segments.size()) {
      // Write the first token.
      text << segments[0];

      // Write the other tokens, each of which is preceded by a delimiter.
      for(size_t k = 1;  k < segments.size();  ++k)
        text << delim << ' ' << segments[k];
    }
    return segments.size();
  }

}  // namespace poseidon
