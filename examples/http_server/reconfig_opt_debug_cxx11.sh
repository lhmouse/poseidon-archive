#!/bin/bash

export CPPFLAGS+=' -D_GLIBCXX_DEBUG -D_GLIBCXX_DEBUG_PEDANTIC -std=c++11'
export CXXFLAGS+=' -O3 -g'

mkdir -p m4
autoreconf -if
./configure --enable-shared --disable-static --prefix=/usr
