#!/bin/bash

autoreconf -if
CPPFLAGS='-DNDEBUG' CFLAGS='-O3 -g -std=c99' CXXFLAGS='-O3 -g -std=c++03'	\
	./configure --enable-shared --disable-static
