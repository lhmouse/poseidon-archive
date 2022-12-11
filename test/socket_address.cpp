// This file is part of Poseidon.
// Copyleft 2022, LH_Mouse. All wrongs reserved.

#include "utils.hpp"
#include "../poseidon/socket/socket_address.hpp"

using namespace ::poseidon;

int main()
  {
    Socket_Address addr;
    POSEIDON_TEST_CHECK(addr.family() == AF_UNSPEC);

    POSEIDON_TEST_CHECK(addr.parse(sref("1.2.128.255"), 12345) == true);
    POSEIDON_TEST_CHECK(addr.family() == AF_INET);
    POSEIDON_TEST_CHECK(::memcmp(&(addr.addr4()->sin_addr), "\x01\x02\x80\xFF", 4) == 0);
    POSEIDON_TEST_CHECK(addr.addr4()->sin_port == htobe16(12345));

    POSEIDON_TEST_CHECK(addr.parse(sref(""), 12345) == true);
    POSEIDON_TEST_CHECK(addr.family() == AF_UNSPEC);
    POSEIDON_TEST_CHECK(addr.parse(sref("12.3.4"), 12345) == false);
    POSEIDON_TEST_CHECK(addr.family() == AF_UNSPEC);

    POSEIDON_TEST_CHECK(addr.parse(sref("[fe80:1234:5678::90ab:cdef]"), 54321) == true);
    POSEIDON_TEST_CHECK(addr.family() == AF_INET6);
    POSEIDON_TEST_CHECK(::memcmp(&(addr.addr6()->sin6_addr), "\xfe\x80\x12\x34\x56\x78\x00\x00\x00\x00\x00\x00\x90\xAB\xCD\xEF", 16) == 0);
    POSEIDON_TEST_CHECK(addr.addr4()->sin_port == htobe16(54321));

    POSEIDON_TEST_CHECK(addr.parse(sref("[::]"), 7890) == true);
    POSEIDON_TEST_CHECK(addr.family() == AF_INET6);
    POSEIDON_TEST_CHECK(::memcmp(&(addr.addr6()->sin6_addr), "\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00", 16) == 0);
    POSEIDON_TEST_CHECK(addr.addr4()->sin_port == htobe16(7890));
  }
