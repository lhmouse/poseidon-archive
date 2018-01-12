#!/bin/bash

_common_pp="-D_GLIBCXX_DEBUG -D_GLIBCXX_DEBUG_PEDANTIC"
_c_pp="-std=c99"
_cxx_pp="-std=c++98"
_optimize="-O0 -g3 -fsanitize=address -fsanitize=leak -fsanitize=undefined"

mkdir -p m4
autoreconf -if

./configure CPPFLAGS="${CPPFLAGS} ${_common_pp} ${_optimize}" CFLAGS="${CFLAGS} ${_c_pp}" CXXFLAGS="${CXXFLAGS} ${_cxx_pp}" LDFLAGS="${LDFLAGS} ${_optimize}"	\
	--enable-shared --disable-static --prefix=/usr/local
