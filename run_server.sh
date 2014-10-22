#!/bin/bash

var=`pwd`'/var'

if [ "$1" == "-d" ]; then
	sudo libtool --mode=execute gdb --args ./sbin/poseidon $var
elif [ "$1" == "-v" ]; then
	sudo libtool --mode=execute valgrind ./sbin/poseidon $var
else
	sudo ./sbin/poseidon $var
fi
