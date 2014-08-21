#!/bin/bash

if [ "$1" == "-d" ]; then
	sudo libtool --mode=execute gdb ./sbin/poseidon
elif [ "$1" == "-v" ]; then
	sudo libtool --mode=execute valgrind ./sbin/poseidon
else
	sudo ./sbin/poseidon
fi
