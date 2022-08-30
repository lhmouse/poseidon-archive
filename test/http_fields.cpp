// This file is part of Poseidon.
// Copyleft 2022, LH_Mouse. All wrongs reserved.

#include "utils.hpp"
#include "../src/http/http_fields.hpp"

using namespace ::poseidon;

int main()
  {
    HTTP_Fields fields;
    POSEIDON_TEST_CHECK(fields.empty() == true);
    POSEIDON_TEST_CHECK(fields.size() == 0);

    fields.reserve(20);
    POSEIDON_TEST_CHECK(fields.empty() == true);
    POSEIDON_TEST_CHECK(fields.size() == 0);
    POSEIDON_TEST_CHECK(fields.capacity() >= 20);

    auto field1 = & fields.append(sref("name1"), sref("value1"));
    POSEIDON_TEST_CHECK(fields.empty() == false);
    POSEIDON_TEST_CHECK(fields.size() == 1);
    POSEIDON_TEST_CHECK(field1->first == "name1");
    POSEIDON_TEST_CHECK(field1->second == "value1");
    POSEIDON_TEST_CHECK(field1 == &*(fields.begin()));
    POSEIDON_TEST_CHECK(field1 == &*(fields.rbegin()));

    auto field2 = & fields.append(sref("name2"), sref("value2"));
    POSEIDON_TEST_CHECK(fields.empty() == false);
    POSEIDON_TEST_CHECK(fields.size() == 2);
    POSEIDON_TEST_CHECK(field2->first == "name2");
    POSEIDON_TEST_CHECK(field2->second == "value2");
    POSEIDON_TEST_CHECK(field1 == &*(fields.begin()));
    POSEIDON_TEST_CHECK(field2 == &*(fields.rbegin()));

    auto field3 = & fields.append(sref("NAME2"), sref("value3"));
    POSEIDON_TEST_CHECK(fields.empty() == false);
    POSEIDON_TEST_CHECK(fields.size() == 3);
    POSEIDON_TEST_CHECK(field3->first == "NAME2");
    POSEIDON_TEST_CHECK(field3->second == "value3");
    POSEIDON_TEST_CHECK(field1 == &*(fields.begin()));
    POSEIDON_TEST_CHECK(field3 == &*(fields.rbegin()));

    POSEIDON_TEST_CHECK(fields.erase(sref("name2")) == 2);
    POSEIDON_TEST_CHECK(fields.empty() == false);
    POSEIDON_TEST_CHECK(fields.size() == 1);
    POSEIDON_TEST_CHECK(field1 == &*(fields.rbegin()));

    POSEIDON_TEST_CHECK(fields.erase(sref("nonexistent")) == 0);
    POSEIDON_TEST_CHECK(fields.empty() == false);
    POSEIDON_TEST_CHECK(fields.size() == 1);

    POSEIDON_TEST_CHECK(fields.erase(sref("name1")) == 1);
    POSEIDON_TEST_CHECK(fields.empty() == true);
    POSEIDON_TEST_CHECK(fields.size() == 0);

    fields.append(sref("name1"), sref("value4"));
    fields.append(sref("name2"), sref("value5"));
    fields.append(sref("NAME2"), sref("value6\nvalue7"));
    ::printf("-- begin 1 --\n%s--- end 1 ---\n", fields.print_to_string().c_str());

    auto cptr = fields.find_opt(sref("NAME1"));
    POSEIDON_TEST_CHECK(cptr->second == "value4");
    cptr = fields.find_opt(sref("NAME2"));
    POSEIDON_TEST_CHECK(cptr->second == "value6\nvalue7");
    cptr = fields.find_opt(sref("nonexistent"));
    POSEIDON_TEST_CHECK(cptr == nullptr);
    POSEIDON_TEST_CHECK(fields.size() == 3);

    auto mptr = fields.mut_find_opt(sref("NAME1"));
    POSEIDON_TEST_CHECK(mptr->second == "value4");
    mptr = fields.mut_find_opt(sref("NAME2"));
    POSEIDON_TEST_CHECK(mptr->second == "value6\nvalue7");
    mptr = fields.mut_find_opt(sref("nonexistent"));
    POSEIDON_TEST_CHECK(mptr == nullptr);
    POSEIDON_TEST_CHECK(fields.size() == 3);

    mptr = fields.squash_opt(sref("NAME1"));
    POSEIDON_TEST_CHECK(mptr->second == "value4");
    mptr = fields.squash_opt(sref("NAME2"));

    POSEIDON_TEST_CHECK(mptr->second == sref("value5, value6\nvalue7"));
    mptr = fields.squash_opt(sref("nonexistent"));
    POSEIDON_TEST_CHECK(mptr == nullptr);
    POSEIDON_TEST_CHECK(fields.size() == 2);

    fields.clear();
    POSEIDON_TEST_CHECK(fields.empty() == true);
    POSEIDON_TEST_CHECK(fields.size() == 0);

    fields.clear();
    fields.append(sref("key"), sref(R"(value with "quotes" and spaces )"));
    fields.append(sref("Domain"), sref("example.com"));
    fields.append_empty(sref("HttpOnly"));
    ::printf("-- begin 3 --\n%s--- end 3 ---\n", fields.print_to_string().c_str());
    POSEIDON_TEST_CHECK(fields.options_encode_as_string() ==
        R"(key="value with \"quotes\" and spaces "; Domain=example.com; HttpOnly)");

    POSEIDON_TEST_CHECK(fields.options_decode(sref(R"(key="value with \"quotes\" and spaces " ; Domain=  example.com  ;HttpOnly)")) == true);
    POSEIDON_TEST_CHECK(fields.size() == 3);
    POSEIDON_TEST_CHECK(fields.at(0).first == "key");
    POSEIDON_TEST_CHECK(fields.at(0).second == R"(value with "quotes" and spaces )");
    POSEIDON_TEST_CHECK(fields.at(1).first == "Domain");
    POSEIDON_TEST_CHECK(fields.at(1).second == "example.com");
    POSEIDON_TEST_CHECK(fields.at(2).first == "HttpOnly");
    POSEIDON_TEST_CHECK(fields.at(2).second == "");

    fields.clear();
    fields.append(sref("abbcc"), sref("abcdАВГД甲乙丙丁z"));
    fields.append(sref("e=g_g"), sref(" \t`~!@#$%^&*()_+-={}|[]\\:\";\'<>?,./"));
    ::printf("-- begin 2 --\n%s--- end 2 ---\n", fields.print_to_string().c_str());
    POSEIDON_TEST_CHECK(fields.query_encode_as_string() ==
        "abbcc=abcd%D0%90%D0%92%D0%93%D0%94%E7%94%B2%E4%B9%99%E4%B8%99%E4%B8%81z&"
        "e%3Dg_g=+%09%60%7E%21%40%23%24%25%5E%26%2A%28%29_%2B-"
        "%3D%7B%7D%7C%5B%5D%5C%3A%22%3B%27%3C%3E%3F%2C.%2F");

    POSEIDON_TEST_CHECK(fields.query_decode(sref("aa=bb=cc&dd&ee=f%20g+h%61&zz")) == true);
    POSEIDON_TEST_CHECK(fields.size() == 4);
    POSEIDON_TEST_CHECK(fields.at(0).first == "aa");
    POSEIDON_TEST_CHECK(fields.at(0).second == "bb=cc");
    POSEIDON_TEST_CHECK(fields.at(1).first == "dd");
    POSEIDON_TEST_CHECK(fields.at(1).second == "");
    POSEIDON_TEST_CHECK(fields.at(2).first == "ee");
    POSEIDON_TEST_CHECK(fields.at(2).second == "f g ha");
    POSEIDON_TEST_CHECK(fields.at(3).first == "zz");
    POSEIDON_TEST_CHECK(fields.at(3).second == "");

    POSEIDON_TEST_CHECK(fields.query_decode(sref("%1gab")) == false);
    POSEIDON_TEST_CHECK(fields.query_decode(sref("ab%3=dd")) == false);
    POSEIDON_TEST_CHECK(fields.query_decode(sref("ab%3&")) == false);
  }
