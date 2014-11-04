#!/bin/bash

var=`pwd`'/var'

if [ "$1" == "-d" ]; then
	libtool --mode=execute gdb --args ./sbin/poseidon $var
elif [ "$1" == "-v" ]; then
	libtool --mode=execute valgrind --leak-check=full --log-file='valgrind.log' ./sbin/poseidon $var
else
	./sbin/poseidon $var
fi
