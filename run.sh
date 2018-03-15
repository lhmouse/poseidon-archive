#!/bin/bash

_etc="$(pwd)/etc/poseidon"

if [ "$1" == "-d" ]; then
	./libtool --mode=execute gdb --args ./bin/poseidon "${_etc}"
elif [ "$1" == "-v" ]; then
	./libtool --mode=execute valgrind --leak-check=full --log-file='valgrind.log' ./bin/poseidon "${_etc}"
elif [ "$1" == "-vgdb" ]; then
	./libtool --mode=execute valgrind --vgdb=yes --vgdb-error=0 --leak-check=full --log-file='valgrind.log' ./bin/poseidon "${_etc}"
else
	./bin/poseidon "${_etc}"
fi
