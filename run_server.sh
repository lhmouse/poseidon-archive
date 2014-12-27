#!/bin/bash

etc=`pwd`'/etc'

if [ "$1" == "-d" ]; then
	libtool --mode=execute gdb --args ./sbin/poseidon $etc/poseidon
elif [ "$1" == "-v" ]; then
	libtool --mode=execute valgrind --leak-check=full --log-file='valgrind.log' ./sbin/poseidon $etc/poseidon
else
	./sbin/poseidon $etc/poseidon
fi
