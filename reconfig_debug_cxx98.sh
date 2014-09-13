#!/bin/bash

mkdir -p m4
autoreconf -if
CPPFLAGS='-D_GLIBCXX_DEBUG' CFLAGS='-O0 -g -std=c99' CXXFLAGS='-O0 -g -std=c++98'	\
	./configure --enable-shared --disable-static
