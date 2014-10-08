#!/bin/bash

if [ "$1" == "-d" ]; then
	sudo libtool --mode=execute gdb ./sbin/poseidon `pwd`/config/conf.rc
elif [ "$1" == "-v" ]; then
	sudo libtool --mode=execute valgrind ./sbin/poseidon `pwd`/config/conf.rc
else
	sudo ./sbin/poseidon `pwd`/config/conf.rc
fi
