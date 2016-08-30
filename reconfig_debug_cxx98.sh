#!/bin/bash

export CPPFLAGS+=' -D_GLIBCXX_DEBUG -D_GLIBCXX_DEBUG_PEDANTIC -std=c++98'
export CXXFLAGS+=' -O0 -g'

mkdir -p m4
autoreconf -if
./configure --enable-shared --disable-static --prefix=/usr/local
