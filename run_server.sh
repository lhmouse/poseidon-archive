#!/bin/bash

etc=`pwd`'/etc'

if [ "$1" == "-d" ]; then
	libtool --mode=execute gdb --args ./bin/poseidon $etc/poseidon
elif [ "$1" == "-v" ]; then
	libtool --mode=execute valgrind --leak-check=full --log-file='valgrind.log' ./bin/poseidon $etc/poseidon
else
	./bin/poseidon $etc/poseidon
fi
