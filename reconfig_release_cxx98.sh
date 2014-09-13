#!/bin/bash

mkdir -p m4
autoreconf -if
CPPFLAGS='-DNDEBUG' CFLAGS='-O3 -g -std=c99' CXXFLAGS='-O3 -g -std=c++98'	\
	./configure --enable-shared --disable-static
