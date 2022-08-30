// This file is part of Poseidon.
// Copyleft 2022, LH_Mouse. All wrongs reserved.

#include "../precompiled.ipp"
#include "http_fields.hpp"
#include "../utils.hpp"

namespace poseidon {
namespace {

void
do_fold_line(tinyfmt& fmt, const cow_string& str)
  {
    auto bpos = str.c_str();
    auto mpos = bpos;
    auto epos = bpos + str.length();

    while(mpos != epos) {
      uint32_t ch = (uint8_t) *(mpos++);

      // If this character should not be escaped, skip it.
      if(ch != '\n')
        continue;

      // If there are pending characters, flush them first.
      if(bpos != mpos - 1)
        fmt.putn(bpos, (size_t) (mpos - 1 - bpos));

      // The character at `mpos[-1]` will be escaped, so move past it.
      bpos = mpos;

      // Insert a HT after the LF.
      fmt.putn("\n\t", 2);
    }

    // If there are pending characters, flush them.
    if(bpos != mpos)
      fmt.putn(bpos, (size_t) (mpos - bpos));
  }

void
do_escape_option(tinyfmt& fmt, const cow_string& str)
  {
    auto bpos = str.c_str();
    auto mpos = bpos;
    auto epos = bpos + str.length();

    char eseq[4] = { "\\" };
    bool quoted = false;

    while(mpos != epos) {
      uint32_t ch = (uint8_t) *(mpos++);

      // Reject control characters.
      if(((ch != '\t') && (ch <= 0x1F)) || (ch == 0x7F))
        POSEIDON_THROW((
            "Character `U+$1` not allowed in HTTP headers"),
            ::rocket::ascii_numput().put_XU(ch, 2));

      // If this character should not be escaped, skip it.
      if((ch != ' ') && (ch != '\\') && (ch != '\"'))
        continue;

      // Open the quote if one hasn't been opened.
      if(!quoted)
        fmt.putc('\"');

      // If there are pending characters, flush them first.
      if(bpos != mpos - 1)
        fmt.putn(bpos, (size_t) (mpos - 1 - bpos));

      // The character at `mpos[-1]` will be escaped, so move past it.
      bpos = mpos;
      quoted = true;

      // Space and horizontal tab characters need no escaping.
      if(is_any_of(ch, { ' ', '\t' })) {
        fmt.putc((char) ch);
        continue;
      }

      // Insert a backslash before this character.
      eseq[1] = (char) ch;
      fmt.putn(eseq, 2);
    }

    // If there are pending characters, flush them.
    if(bpos != mpos)
      fmt.putn(bpos, (size_t) (mpos - bpos));

    // Close the quote, if any.
    if(quoted)
      fmt.putc('\"');
  }

void
do_urlencode(tinyfmt& fmt, const cow_string& str)
  {
    auto bpos = str.c_str();
    auto mpos = bpos;
    auto epos = bpos + str.length();

    char eseq[4] = { "%" };
    uint32_t digit;

    while(mpos != epos) {
      uint32_t ch = (uint8_t) *(mpos++);

      // If this character should not be encoded, skip it.
      if((ch >= '0') && (ch <= '9'))
        continue;

      if(((ch | 0x20) >= 'a') && ((ch | 0x20) <= 'z'))
        continue;

      if((ch == '-') || (ch == '_') || (ch == '.'))
        continue;

      // If there are pending characters, flush them first.
      if(bpos != mpos - 1)
        fmt.putn(bpos, (size_t) (mpos - 1 - bpos));

      // The character at `mpos[-1]` will be encoded, so move past it.
      bpos = mpos;

      // Encode spaces specially.
      if(ch == ' ') {
        fmt.putc('+');
        continue;
      }

      // Encode this character.
      digit = '0' + ch / 16U;
      eseq[1] = (char) (digit + (('9' - digit) >> 29));
      digit = '0' + ch % 16U;
      eseq[2] = (char) (digit + (('9' - digit) >> 29));
      fmt.putn(eseq, 3);
    }

    // If there are pending characters, flush them.
    if(bpos != mpos)
      fmt.putn(bpos, (size_t) (mpos - bpos));
  }

}  // namespace

HTTP_Fields::
~HTTP_Fields()
  {
  }

HTTP_Fields&
HTTP_Fields::
reserve(size_t res_arg)
  {
    this->m_stor.reserve(res_arg);
    return *this;
  }

HTTP_Fields&
HTTP_Fields::
shrink_to_fit()
  {
    this->m_stor.shrink_to_fit();
    return *this;
  }

HTTP_Fields&
HTTP_Fields::
clear() noexcept
  {
    this->m_stor.clear();
    return *this;
  }

HTTP_Fields::iterator
HTTP_Fields::
insert(const_iterator pos, const value_type& field)
  {
    return this->m_stor.insert(pos, field);
  }

HTTP_Fields::iterator
HTTP_Fields::
insert(const_iterator pos, value_type&& field)
  {
    return this->m_stor.insert(pos, ::std::move(field));
  }

HTTP_Fields::iterator
HTTP_Fields::
erase(const_iterator pos)
  {
    return this->m_stor.erase(pos);
  }

HTTP_Fields::iterator
HTTP_Fields::
erase(const_iterator first, const_iterator last)
  {
    return this->m_stor.erase(first, last);
  }

tinyfmt&
HTTP_Fields::
print(tinyfmt& fmt) const
  {
    for(const auto& field : this->m_stor) {
      // Write the key-value pair. No validation is performed.
      do_fold_line(fmt, field.first);
      fmt.putn(": ", 2);
      do_fold_line(fmt, field.second);
      fmt.putc('\n');
    }
    return fmt;
  }

cow_string
HTTP_Fields::
print_to_string() const
  {
    ::rocket::tinyfmt_str fmt;
    this->print(fmt);
    return fmt.extract_string();
  }

bool
HTTP_Fields::
parse(const cow_string& lines)
  {
    this->m_stor.clear();
/*
    // TODO

    // An empty string denotes an empty set of values.
    if(ROCKET_UNEXPECT(lines.empty()))
      return true;

    value_type* cur_value = nullptr;
    cow_string* cur_sink = nullptr;

    auto bpos = text.c_str();
    auto mpos = bpos;
    auto epos = bpos + text.length();

    while(mpos != epos) {
      uint32_t ch = (uint8_t) *(mpos++);

      // Check for line termination.
      if(ch == '\n') {
        if(mpos == epos)
          break;

        if((*mpos == ' ') || (*mpos == '\t')) {



      // If this character is a line break, terminate the current field.
      if(ch == '\n') {
        cur_value = nullptr;
        cur_sink = nullptr;
        continue;
      }

      // Otherwise, it is part of a field, so create one as needed.
      if(!cur_value) {
        cur_value = &(this->m_stor.emplace_back());
        cur_sink = &(cur_value->first);
      }

      // If it is an equals sign and we are decoding the name, stop
      // to decode the value now.
      if((ch == '=') && (cur_sink == &(cur_value->first))) {
        cur_sink = &(cur_value->second);
        continue;
      }

      // Reject control and blank characters.
      if((ch <= 0x20) || (ch == 0x7F))
        return false;

      // Decode spaces specially.
      if(ch == '+') {
        cur_sink->push_back(' ');
        continue;
      }

      // Accept unencoded characters verbatim.
      if(ch != '%') {
        cur_sink->push_back((char) ch);
        continue;
      }

      // Get two consecutive hexadecimal digits.
      if(epos - mpos < 2)
        return false;

      if((mpos[0] >= '0') && (mpos[0] <= '9'))
        ch = (uint32_t) (mpos[0] - '0') * 16U;
      else if(((mpos[0] | 0x20) >= 'a') && ((mpos[0] | 0x20) <= 'f'))
        ch = (uint32_t) ((mpos[0] | 0x20) - 'a' + 10) * 16U;
      else
        return false;

      if((mpos[1] >= '0') && (mpos[1] <= '9'))
        ch += (uint32_t) (mpos[1] - '0');
      else if(((mpos[1] | 0x20) >= 'a') && ((mpos[1] | 0x20) <= 'f'))
        ch += (uint32_t) ((mpos[1] | 0x20) - 'a' + 10);
      else
        return false;

      // Accept this byte.
      cur_sink->push_back((char) ch);
      mpos += 2;
    }
*/
    return true;
  }

tinyfmt&
HTTP_Fields::
options_encode(tinyfmt& fmt) const
  {
    for(const auto& field : this->m_stor) {
      // If this is not the first field, insert a separator.
      if(&field != this->m_stor.data())
        fmt.putn("; ", 2);

      // Write the key-value pair, separated by an equals sign.
      do_escape_option(fmt, field.first);

      if(!field.second.empty())
        fmt.putc('='),
          do_escape_option(fmt, field.second);
    }
    return fmt;
  }

cow_string
HTTP_Fields::
options_encode_as_string() const
  {
    ::rocket::tinyfmt_str fmt;
    this->options_encode(fmt);
    return fmt.extract_string();
  }

bool
HTTP_Fields::
options_decode(const cow_string& text)
  {
    this->m_stor.clear();

    // An empty string denotes an empty set of values.
    if(ROCKET_UNEXPECT(text.empty()))
      return true;

    enum Parser_State
      {
        parser_l_space,  // leading space; no character accepted
        parser_uq_text,  // unquoted text
        parser_t_space,  // trailing space
        parser_quo_in,   // inside quotes
        parser_quo_esc,  // inside quotes; escaped
        parser_quo_end,  // end of quoted string
      };

    value_type* cur_value = nullptr;
    cow_string* cur_sink = nullptr;
    Parser_State cur_state = parser_l_space;

    auto bpos = text.c_str();
    auto mpos = bpos;
    auto epos = bpos + text.length();

    while(mpos != epos) {
      uint32_t ch = (uint8_t) *(mpos++);

      // If this character is a separator, terminate the current field.
      if(ch == ';') {
        cur_value = nullptr;
        cur_sink = nullptr;
        cur_state = parser_l_space;
        continue;
      }

      // Reject control characters.
      if(((ch != '\t') && (ch <= 0x1F)) || (ch == 0x7F))
        return false;

      // When this character is blank, but no key has begun, ignore it.
      if(is_any_of(cur_state, { parser_l_space, parser_t_space, parser_quo_end }) && is_any_of(ch, { ' ', '\t' }))
        continue;

      // Otherwise, it is part of a field, so create one as needed.
      if(!cur_value) {
        cur_value = &(this->m_stor.emplace_back());
        cur_sink = &(cur_value->first);
      }

      // If it is a double quotation mark, which should not be part
      // of the value, skip it.
      if((cur_state == parser_l_space) && (ch == '\"')) {
        cur_state = parser_quo_in;
        continue;
      }

      // Escape sequences are allowed in quoted text.
      if((cur_state == parser_quo_in) && (ch == '\\')) {
        cur_state = parser_quo_esc;
        continue;
      }

      if((cur_state == parser_quo_in) && (ch == '\"')) {
        cur_state = parser_quo_end;
        continue;
      }

      if((cur_state == parser_quo_in) || (cur_state == parser_quo_esc)) {
        cur_sink->push_back((char) ch);
        cur_state = parser_quo_in;
        continue;
      }

      // Only blank characters are allowed to follow quoted text.
      if(cur_state == parser_quo_end)
        return false;

      // If it is an equals sign and we are decoding the name, stop
      // to decode the value now.
      if((ch == '=') && (cur_sink == &(cur_value->first))) {
        cur_sink = &(cur_value->second);
        cur_state = parser_l_space;
        continue;
      }

      // Handle trailing spaces.
      if(is_any_of(ch, { ' ', '\t' })) {
        cur_state = parser_t_space;
        continue;
      }

      // If trailing space characters have been accepted previously,
      // squash them here.
      if(cur_state == parser_t_space)
        cur_sink->push_back(' ');

      // Accept this character.
      cur_sink->push_back((char) ch);
      cur_state = parser_uq_text;
    }
    return true;
  }

tinyfmt&
HTTP_Fields::
query_encode(tinyfmt& fmt) const
  {
    for(const auto& field : this->m_stor) {
      // If this is not the first field, insert a separator.
      if(&field != this->m_stor.data())
        fmt.putc('&');

      // Write the key-value pair, separated by an equals sign.
      do_urlencode(fmt, field.first);
      fmt.putc('=');
      do_urlencode(fmt, field.second);
    }
    return fmt;
  }

cow_string
HTTP_Fields::
query_encode_as_string() const
  {
    ::rocket::tinyfmt_str fmt;
    this->query_encode(fmt);
    return fmt.extract_string();
  }

bool
HTTP_Fields::
query_decode(const cow_string& text)
  {
    this->m_stor.clear();

    // An empty string denotes an empty set of values.
    if(ROCKET_UNEXPECT(text.empty()))
      return true;

    value_type* cur_value = nullptr;
    cow_string* cur_sink = nullptr;

    auto mpos = text.c_str();
    auto epos = mpos + text.length();

    while(mpos != epos) {
      uint32_t ch = (uint8_t) *(mpos++);

      // If this character is a separator, terminate the current field.
      if(ch == '&') {
        cur_value = nullptr;
        cur_sink = nullptr;
        continue;
      }

      // Reject control and blank characters.
      if((ch <= 0x20) || (ch == 0x7F))
        return false;

      // Otherwise, it is part of a field, so create one as needed.
      if(!cur_value) {
        cur_value = &(this->m_stor.emplace_back());
        cur_sink = &(cur_value->first);
      }

      // If it is an equals sign and we are decoding the name, stop
      // to decode the value now.
      if((ch == '=') && (cur_sink == &(cur_value->first))) {
        cur_sink = &(cur_value->second);
        continue;
      }

      // Decode spaces specially.
      if(ch == '+') {
        cur_sink->push_back(' ');
        continue;
      }

      // Accept unencoded characters verbatim.
      if(ch != '%') {
        cur_sink->push_back((char) ch);
        continue;
      }

      // Get two consecutive hexadecimal digits.
      if(epos - mpos < 2)
        return false;

      if((mpos[0] >= '0') && (mpos[0] <= '9'))
        ch = (uint32_t) (mpos[0] - '0') * 16U;
      else if(((mpos[0] | 0x20) >= 'a') && ((mpos[0] | 0x20) <= 'f'))
        ch = (uint32_t) ((mpos[0] | 0x20) - 'a' + 10) * 16U;
      else
        return false;

      if((mpos[1] >= '0') && (mpos[1] <= '9'))
        ch += (uint32_t) (mpos[1] - '0');
      else if(((mpos[1] | 0x20) >= 'a') && ((mpos[1] | 0x20) <= 'f'))
        ch += (uint32_t) ((mpos[1] | 0x20) - 'a' + 10);
      else
        return false;

      // Accept this byte.
      cur_sink->push_back((char) ch);
      mpos += 2;
    }
    return true;
  }

const HTTP_Fields::value_type*
HTTP_Fields::
find_opt(const cow_string& name) const noexcept
  {
    const value_type* ptr = nullptr;
    auto cur = this->m_stor.end();

    // Look for matches backwards. Keys are case-insenstive.
    while(cur != this->m_stor.begin()) {
      const auto& field = *--cur;
      if(ascii_ci_equal(field.first, name)) {
        // Use this field.
        ptr = &field;
        break;
      }
    }
    return ptr;
  }

HTTP_Fields::value_type*
HTTP_Fields::
mut_find_opt(const cow_string& name) noexcept
  {
    value_type* ptr = nullptr;
    auto cur = this->m_stor.end();

    // Look for matches backwards. Keys are case-insenstive.
    while(cur != this->m_stor.begin()) {
      const auto& field = *--cur;
      if(ascii_ci_equal(field.first, name)) {
        // Use this field.
        ptrdiff_t off = cur - this->m_stor.begin();
        ptr = this->m_stor.mut_data() + off;
        break;
      }
    }
    return ptr;
  }

HTTP_Fields::value_type*
HTTP_Fields::
squash_opt(const cow_string& name)
  {
    value_type* ptr = nullptr;
    auto cur = this->m_stor.end();

    // Look for matches backwards. Keys are case-insenstive.
    while(cur != this->m_stor.begin()) {
      const auto& field = *--cur;
      if(ascii_ci_equal(field.first, name)) {
        // Use this field.
        ptrdiff_t off = cur - this->m_stor.begin();
        value_type* old = ptr;
        ptr = this->m_stor.mut_data() + off;

        if(old) {
          // If a previous match exists, concatenate their values.
          ptr->second << ", " << old->second;
          this->m_stor.erase(cur + (old - ptr));
        }
      }
    }
    return ptr;
  }

HTTP_Fields::value_type&
HTTP_Fields::
append(const value_type& field)
  {
    return this->m_stor.emplace_back(field);
  }

HTTP_Fields::value_type&
HTTP_Fields::
append(value_type&& field)
  {
    return this->m_stor.emplace_back(::std::move(field));
  }

HTTP_Fields::value_type&
HTTP_Fields::
append(const cow_string& name, const cow_string& value)
  {
    return this->m_stor.emplace_back(name, value);
  }

HTTP_Fields::value_type&
HTTP_Fields::
append_empty(const cow_string& name)
  {
    return this->m_stor.emplace_back(::std::piecewise_construct, ::std::tie(name), ::std::tie());
  }

size_t
HTTP_Fields::
erase(const cow_string& name)
  {
    size_t nerased = 0;
    auto cur = this->m_stor.end();

    // Look for matches backwards. Keys are case-insenstive.
    while(cur != this->m_stor.begin()) {
      const auto& field = *--cur;
      if(ascii_ci_equal(field.first, name)) {
        // Erase this field.
        cur = this->m_stor.erase(cur);
        nerased ++;
      }
    }
    return nerased;
  }

}  // namespace poseidon
