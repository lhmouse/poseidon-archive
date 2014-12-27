#!/bin/bash

etc=`pwd`'/etc'

if [ "$1" == "-d" ]; then
	libtool --mode=execute gdb --args ./sbin/poseidon $etc
elif [ "$1" == "-v" ]; then
	libtool --mode=execute valgrind --leak-check=full --log-file='valgrind.log' ./sbin/poseidon $etc
else
	./sbin/poseidon $etc
fi
