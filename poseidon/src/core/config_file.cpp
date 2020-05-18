// This file is part of Poseidon.
// Copyleft 2020, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "config_file.hpp"
#include "../utilities.hpp"
#include <asteria/compiler/token_stream.hpp>
#include <asteria/compiler/enums.hpp>
#include <asteria/compiler/parser_error.hpp>
#include <stdlib.h>  // ::realpath()

namespace poseidon {
namespace {

inline
bool
do_check_punctuator(const ::asteria::Token* qtok, initializer_list<::asteria::Punctuator> accept)
  {
    return qtok && qtok->is_punctuator() && ::rocket::is_any_of(qtok->as_punctuator(), accept);
  }

struct Key_with_sloc
  {
    ::asteria::Source_Location sloc;
    size_t length;
    phsh_string name;
  };

opt<Key_with_sloc>
do_accept_object_key_opt(::asteria::Token_Stream& tstrm)
  {
    auto qtok = tstrm.peek_opt();
    if(!qtok)
      return nullopt;

    // A key may be either an identifier or a string literal.
    Key_with_sloc key;
    if(qtok->is_identifier()) {
      key.name = qtok->as_identifier();
    }
    else if(qtok->is_string_literal()) {
      key.name = qtok->as_string_literal();
    }
    else {
      return nullopt;
    }
    key.sloc = qtok->sloc();
    key.length = qtok->length();
    tstrm.shift();

    // Accept the value initiator.
    qtok = tstrm.peek_opt();
    if(!do_check_punctuator(qtok, { ::asteria::punctuator_assign, ::asteria::punctuator_colon }))
      throw ::asteria::Parser_Error(::asteria::parser_status_equals_sign_or_colon_expected,
                                    tstrm.next_sloc(), tstrm.next_length());
    tstrm.shift();

    return ::std::move(key);
  }

::asteria::Value&
do_insert_unique(::asteria::V_object& obj, Key_with_sloc&& key, ::asteria::Value&& value)
  {
    auto pair = obj.try_emplace(::std::move(key.name), ::std::move(value));
    if(!pair.second)
      throw ::asteria::Parser_Error(::asteria::parser_status_duplicate_key_in_object,
                                    key.sloc, key.length);
    return pair.first->second;
  }

struct S_xparse_array
  {
    ::asteria::V_array arr;
  };

struct S_xparse_object
  {
    ::asteria::V_object obj;
    Key_with_sloc key;
  };

using Xparse = ::rocket::variant<S_xparse_array, S_xparse_object>;

::asteria::Value
do_parse_value_nonrecursive(::asteria::Token_Stream& tstrm)
  {
    ::asteria::Value value;

    // Implement a non-recursive descent parser.
    cow_vector<Xparse> stack;

    for(;;) {
      // Accept a value. No other things such as closed brackets are allowed.
      auto qtok = tstrm.peek_opt();
      if(!qtok)
        throw ::asteria::Parser_Error(::asteria::parser_status_expression_expected,
                                      tstrm.next_sloc(), tstrm.next_length());

      switch(weaken_enum(qtok->index())) {
        case ::asteria::Token::index_punctuator: {
          // Accept a `+`, `-`, `[` or `{`.
          auto punct = qtok->as_punctuator();
          switch(weaken_enum(punct)) {
            case ::asteria::punctuator_add:
            case ::asteria::punctuator_sub: {
              cow_string name;
              qtok = tstrm.peek_opt(1);
              if(qtok && qtok->is_identifier())
                name = qtok->as_identifier();

              // Only `Infinity` and `NaN` may follow.
              // Note that the tokenizer will have merged sign symbols into adjacent number literals.
              if(::rocket::is_none_of(name, { "Infinity", "NaN" }))
                throw ::asteria::Parser_Error(::asteria::parser_status_expression_expected,
                                              tstrm.next_sloc(), tstrm.next_length());

              // Accept a special numeric value.
              double sign = (punct == ::asteria::punctuator_add) - 1;
              double real = (name[0] == 'I') ? ::std::numeric_limits<double>::infinity()
                                             : ::std::numeric_limits<double>::quiet_NaN();

              value = ::std::copysign(real, sign);
              tstrm.shift(2);
              break;
            }

            case ::asteria::punctuator_bracket_op: {
              tstrm.shift();

              // Open an array.
              qtok = tstrm.peek_opt();
              if(!qtok) {
                throw ::asteria::Parser_Error(::asteria::parser_status_closed_bracket_or_comma_expected,
                                              tstrm.next_sloc(), tstrm.next_length());
              }
              else if(!do_check_punctuator(qtok, { ::asteria::punctuator_bracket_cl })) {
                // Descend into the new array.
                S_xparse_array ctxa = { ::asteria::V_array() };
                stack.emplace_back(::std::move(ctxa));
                continue;
              }
              tstrm.shift();

              // Accept an empty array.
              value = ::asteria::V_array();
              break;
            }

            case ::asteria::punctuator_brace_op: {
              tstrm.shift();

              // Open an object.
              qtok = tstrm.peek_opt();
              if(!qtok) {
                throw ::asteria::Parser_Error(::asteria::parser_status_closed_brace_or_comma_expected,
                                              tstrm.next_sloc(), tstrm.next_length());
              }
              else if(!do_check_punctuator(qtok, { ::asteria::punctuator_brace_cl })) {
                // Get the first key.
                auto qkey = do_accept_object_key_opt(tstrm);
                if(!qkey)
                  throw ::asteria::Parser_Error(::asteria::parser_status_closed_brace_or_json5_key_expected,
                                                tstrm.next_sloc(), tstrm.next_length());

                // Descend into the new object.
                S_xparse_object ctxo = { ::asteria::V_object(), ::std::move(*qkey) };
                stack.emplace_back(::std::move(ctxo));
                continue;
              }
              tstrm.shift();

              // Accept an empty object.
              value = ::asteria::V_object();
              break;
            }

            default:
              throw ::asteria::Parser_Error(::asteria::parser_status_expression_expected,
                                            tstrm.next_sloc(), tstrm.next_length());
          }
          break;
        }

        case ::asteria::Token::index_identifier: {
          // Accept a literal.
          const auto& name = qtok->as_identifier();
          if(::rocket::is_none_of(name, { "null", "true", "false", "Infinity", "NaN" }))
            throw ::asteria::Parser_Error(::asteria::parser_status_expression_expected,
                                          tstrm.next_sloc(), tstrm.next_length());

          switch(name[0]) {
            case 'n':
              value = nullptr;
              break;

            case 't':
              value = true;
              break;

            case 'f':
              value = false;
              break;

            case 'I':
              value = ::std::numeric_limits<double>::infinity();
              break;

            case 'N':
              value = ::std::numeric_limits<double>::quiet_NaN();
              break;

            default:
              ROCKET_ASSERT(false);
          }
          tstrm.shift();
          break;
        }

        case ::asteria::Token::index_integer_literal:
          // Accept an integer.
          value = qtok->as_integer_literal();
          tstrm.shift();
          break;

        case ::asteria::Token::index_real_literal:
          // Accept a real.
          value = qtok->as_real_literal();
          tstrm.shift();
          break;

        case ::asteria::Token::index_string_literal:
          // Accept a UTF-8 string.
          value = qtok->as_string_literal();
          tstrm.shift();
          break;

        default:
          throw ::asteria::Parser_Error(::asteria::parser_status_expression_expected,
                                        tstrm.next_sloc(), tstrm.next_length());
      }

      // A complete value has been accepted. Insert it into its parent array or object.
      for(;;) {
        if(stack.empty())
          // Accept the root value.
          return value;

        if(stack.back().index() == 0) {
          auto& ctxa = stack.mut_back().as<0>();
          ctxa.arr.emplace_back(::std::move(value));

          // Check for termination.
          qtok = tstrm.peek_opt();
          if(!qtok) {
            throw ::asteria::Parser_Error(::asteria::parser_status_closed_bracket_or_comma_expected,
                                          tstrm.next_sloc(), tstrm.next_length());
          }
          else if(!do_check_punctuator(qtok, { ::asteria::punctuator_bracket_cl })) {
            // Look for the next element.
            break;
          }
          tstrm.shift();

          // Close this array.
          value = ::std::move(ctxa.arr);
        }
        else {
          auto& ctxo = stack.mut_back().as<1>();
          do_insert_unique(ctxo.obj, ::std::move(ctxo.key), ::std::move(value));

          // Check for termination.
          qtok = tstrm.peek_opt();
          if(!qtok) {
            throw ::asteria::Parser_Error(::asteria::parser_status_closed_brace_or_comma_expected,
                                          tstrm.next_sloc(), tstrm.next_length());
          }
          else if(!do_check_punctuator(qtok, { ::asteria::punctuator_brace_cl })) {
            // Get the next key.
            auto qkey = do_accept_object_key_opt(tstrm);
            if(!qkey)
              throw ::asteria::Parser_Error(::asteria::parser_status_closed_brace_or_json5_key_expected,
                                            tstrm.next_sloc(), tstrm.next_length());

            // Look for the next value.
            ctxo.key = ::std::move(*qkey);
            break;
          }
          tstrm.shift();

          // Close this object.
          value = ::std::move(ctxo.obj);
        }
        stack.pop_back();
      }
    }
  }

}  // namespace

Config_File::
~Config_File()
  {
  }

const ::asteria::Value&
Config_File::
do_throw_type_mismatch(const char* const* bptr, size_t epos, const char* expect,
                       const ::asteria::Value& value)
const
  {
    // Compose the path.
    cow_string path;
    path << '`';
    ::std::for_each(bptr, bptr + epos, [&](const char* s) { path << s << '.';  });
    path.mut_back() = '`';

    // Throw the exception now.
    POSEIDON_THROW("unexpected type of $1 (expecting $2, got `$3`)\n"
                   "[in file '$4']",
                   path, expect, ::asteria::describe_vtype(value.vtype()),
                   this->m_abspath);
  }

Config_File&
Config_File::
reload(const char* path)
  {
    // Resolve the path to an absolute one.
    uptr<char, void (&)(void*)> abspath(::realpath(path, nullptr), ::free);
    if(!abspath)
      POSEIDON_THROW("could not open config file '$2'\n"
                     "[`realpath()` failed: $1]",
                     format_errno(errno), path);

    unique_posix_file fp(::fopen(abspath, "r"), ::fclose);
    if(!fp)
      POSEIDON_THROW("could not open config file '$2'\n"
                     "[`fopen()` failed: $1]",
                     format_errno(errno), abspath);

    // Initialize.
    this->m_abspath.assign(abspath.get());
    this->m_root.clear();

    // Parse characters from the file.
    ::setbuf(fp, nullptr);
    ::rocket::tinybuf_file cbuf(::std::move(fp));

    // Initialize tokenizer options.
    // Unlike JSON5, we support _real_ integers and single-quote string literals.
    ::asteria::Compiler_Options opts;
    opts.keywords_as_identifiers = true;

    ::asteria::Token_Stream tstrm(opts);
    tstrm.reload(cbuf, this->m_abspath);

    // Parse a sequence of key-value pairs.
    while(auto qkey = do_accept_object_key_opt(tstrm))
      do_insert_unique(this->m_root, ::std::move(*qkey),
                       do_parse_value_nonrecursive(tstrm));

    // Ensure all data have been consumed.
    if(!tstrm.empty())
      throw ::asteria::Parser_Error(::asteria::parser_status_identifier_expected,
                                    tstrm.next_sloc(), tstrm.next_length());

    return *this;
  }

const ::asteria::Value&
Config_File::
get_value(const char* const* psegs, size_t nsegs)
const
  {
    auto qobj = ::rocket::ref(this->m_root);
    size_t icur = 0;
    if(icur == nsegs)
      POSEIDON_THROW("empty path not valid");

    for(;;) {
      // Find the child denoted by `*sptr`.
      // Return null if no such child exists or if an explicit null is found.
      auto qchild = qobj->get_ptr(::rocket::sref(psegs[icur]));
      if(!qchild || qchild->is_null())
        return ::asteria::null_value;

      // Advance to the next segment.
      // If the end of `path` is reached, we are done.
      if(++icur == nsegs)
        return *qchild;

      // If more segments follow, the child must be an object.
      if(!qchild->is_object())
        do_throw_type_mismatch(psegs, icur, "`object`", *qchild);

      qobj = ::rocket::ref(qchild->as_object());
    }
  }

opt<bool>
Config_File::
get_bool_opt(const char* const* psegs, size_t nsegs)
const
  {
    const auto& value = this->get_value(psegs, nsegs);
    if(value.is_null())
      return nullopt;

    if(!value.is_boolean())
      do_throw_type_mismatch(psegs, nsegs, "`boolean`", value);

    return value.as_boolean();
  }

opt<int64_t>
Config_File::
get_int64_opt(const char* const* psegs, size_t nsegs)
const
  {
    const auto& value = this->get_value(psegs, nsegs);
    if(value.is_null())
      return nullopt;

    if(!value.is_integer())
      do_throw_type_mismatch(psegs, nsegs, "`integer`", value);

    return value.as_integer();
  }

opt<double>
Config_File::
get_double_opt(const char* const* psegs, size_t nsegs)
const
  {
    const auto& value = this->get_value(psegs, nsegs);
    if(value.is_null())
      return nullopt;

    if(!value.is_convertible_to_real())
      do_throw_type_mismatch(psegs, nsegs, "`integer` or `real`", value);

    return value.convert_to_real();
  }

opt<cow_string>
Config_File::
get_string_opt(const char* const* psegs, size_t nsegs)
const
  {
    const auto& value = this->get_value(psegs, nsegs);
    if(value.is_null())
      return nullopt;

    if(!value.is_string())
      do_throw_type_mismatch(psegs, nsegs, "`string`", value);

    return value.as_string();
  }

opt<::asteria::V_array>
Config_File::
get_array_opt(const char* const* psegs, size_t nsegs)
const
  {
    const auto& value = this->get_value(psegs, nsegs);
    if(value.is_null())
      return nullopt;

    if(!value.is_array())
      do_throw_type_mismatch(psegs, nsegs, "`array`", value);

    return value.as_array();
  }

opt<::asteria::V_object>
Config_File::
get_object_opt(const char* const* psegs, size_t nsegs)
const
  {
    const auto& value = this->get_value(psegs, nsegs);
    if(value.is_null())
      return nullopt;

    if(!value.is_object())
      do_throw_type_mismatch(psegs, nsegs, "`object`", value);

    return value.as_object();
  }

}  // namespace poseidon
