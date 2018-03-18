#!/bin/bash

set -e

_common_pp="-DNDEBUG"
_c_pp="-std=c99"
_cxx_pp="-std=c++98"
_optimize="-O2 -g3"

mkdir -p m4
autoreconf -if

./configure CPPFLAGS="${CPPFLAGS} ${_common_pp} ${_optimize}" CFLAGS="${CFLAGS} ${_c_pp}" CXXFLAGS="${CXXFLAGS} ${_cxx_pp}" LDFLAGS="${LDFLAGS} ${_optimize}"	\
	--enable-shared --disable-static --prefix="/usr/local" $*
