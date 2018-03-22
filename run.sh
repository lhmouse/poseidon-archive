#!/bin/bash

set -e

_type="$1"
test -z "${_type}" || shift

_runpath="$(find $(pwd) -path '*/lib/.libs' -type d -print0 | sed -r 's/\x00/:/g')"
_confpath="$(pwd)/etc/poseidon"

export LD_LIBRARY_PATH="${_runpath}"

if [[ "{_type}" == "-d" ]]; then
	./libtool --mode=execute gdb --args poseidon "${_confpath}" $*
elif [[ "{_type}" == "-v" ]]; then
	./libtool --mode=execute valgrind --leak-check=full --log-file='valgrind.log' poseidon "${_confpath}" $*
elif [[ "{_type}" == "-vgdb" ]]; then
	./libtool --mode=execute valgrind --vgdb=yes --vgdb-error=0 --leak-check=full --log-file='valgrind.log' poseidon "${_confpath}" $*
else
	poseidon "${_confpath}" $*
fi
