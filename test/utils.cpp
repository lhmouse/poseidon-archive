// This file is part of Poseidon.
// Copyleft 2022, LH_Mouse. All wrongs reserved.

#include "utils.hpp"
#include "../src/utils.hpp"

using namespace ::poseidon;

int main()
  {
    try {
      POSEIDON_THROW((
          "test $1 $2 $$/end"),
          "exception:", 42);
    }
    catch(exception& e) {
      POSEIDON_TEST_CHECK(::std::strstr(e.what(),
          "test exception: 42 $/end") != nullptr);
    }
  }
