#!/bin/bash

if [ "$1" == "-d" ]; then
	sudo libtool --mode=execute gdb --args ./sbin/poseidon `pwd`/config/poseidon.conf
elif [ "$1" == "-v" ]; then
	sudo libtool --mode=execute valgrind ./sbin/poseidon `pwd`/config/poseidon.conf
else
	sudo ./sbin/poseidon `pwd`/config/poseidon.conf
fi
