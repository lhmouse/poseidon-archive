// This file is part of Poseidon.
// Copyleft 2022, LH_Mouse. All wrongs reserved.

#include "utils.hpp"
#include "../poseidon/socket/socket_address.hpp"

using namespace ::poseidon;

int main()
  {
    Socket_Address addr;

    POSEIDON_TEST_CHECK(addr.parse(sref("1.2.128.255:12345")) == true);
    POSEIDON_TEST_CHECK(::memcmp(&(addr.data()),
        "\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xFF\xFF\x01\x02\x80\xFF", 16) == 0);
    POSEIDON_TEST_CHECK(addr.port() == 12345);

    POSEIDON_TEST_CHECK(addr.parse(sref("3.2.128.255")) == false);
    POSEIDON_TEST_CHECK(::memcmp(&(addr.data()),
        "\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00", 16) == 0);
    POSEIDON_TEST_CHECK(addr.port() == 0);

    POSEIDON_TEST_CHECK(addr.parse(sref("")) == true);
    POSEIDON_TEST_CHECK(::memcmp(&(addr.data()),
        "\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00", 16) == 0);
    POSEIDON_TEST_CHECK(addr.port() == 0);

    POSEIDON_TEST_CHECK(addr.parse(sref("[fe80:1234:5678::90ab:cdef]:54321")) == true);
    POSEIDON_TEST_CHECK(::memcmp(&(addr.data()),
        "\xfe\x80\x12\x34\x56\x78\x00\x00\x00\x00\x00\x00\x90\xAB\xCD\xEF", 16) == 0);
    POSEIDON_TEST_CHECK(addr.port() == 54321);

    POSEIDON_TEST_CHECK(addr.parse(sref("[::]:7890")) == true);
    POSEIDON_TEST_CHECK(::memcmp(&(addr.data()),
        "\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00", 16) == 0);
    POSEIDON_TEST_CHECK(addr.port() == 7890);

    POSEIDON_TEST_CHECK(addr.parse(sref("[fe80:1254:5678::90ab:cdef]:0")) == true);
    POSEIDON_TEST_CHECK(::memcmp(&(addr.data()),
        "\xfe\x80\x12\x54\x56\x78\x00\x00\x00\x00\x00\x00\x90\xAB\xCD\xEF", 16) == 0);
    POSEIDON_TEST_CHECK(addr.port() == 0);
  }
