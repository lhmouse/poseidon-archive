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

Config_File&
Config_File::
reload(const char* path)
  {
    // Resolve the path to an absolute one.
    uptr<char, void (&)(void*)> rpath(::realpath(path, nullptr), ::free);
    if(!rpath)
      ASTERIA_THROW_SYSTEM_ERROR("realpath");
    cow_string abspath(rpath);

    ::rocket::unique_posix_file fp(::fopen(rpath, "r"), ::fclose);
    if(!fp)
      ASTERIA_THROW_SYSTEM_ERROR("fopen");
    ::setbuf(fp, nullptr);

    // Parse characters from the file.
    ::rocket::tinybuf_file cbuf(::std::move(fp));

    // Initialize tokenizer options.
    // Unlike JSON5, we support _real_ integers and single-quote string literals.
    ::asteria::Compiler_Options opts;
    opts.keywords_as_identifiers = true;

    ::asteria::Token_Stream tstrm(opts);
    tstrm.reload(cbuf, abspath);

    // Parse a sequence of key-value pairs.
    ::asteria::V_object root;
    while(auto qkey = do_accept_object_key_opt(tstrm))
      do_insert_unique(root, ::std::move(*qkey), do_parse_value_nonrecursive(tstrm));

    // Ensure all data have been consumed.
    if(!tstrm.empty())
      throw ::asteria::Parser_Error(::asteria::parser_status_identifier_expected,
                                    tstrm.next_sloc(), tstrm.next_length());

    // Accept the value.
    this->m_abspath = ::std::move(abspath);
    this->m_root = ::std::move(root);
    return *this;
  }

}  // namespace poseidon
