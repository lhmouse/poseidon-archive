#!/bin/bash

set -e

_pwd="$(pwd)"
_cmd="${_pwd}/bin/poseidon"
_runpath="$(find "${_pwd}" -path '*/lib/.libs' -type d -print0 | sed -r 's/\x00/:/g')"
_confpath="${_pwd}/etc/poseidon"

_type="$1"
[[ "$#" -ge 1 ]] && shift

export LD_LIBRARY_PATH="${_runpath}"

case "${_type}" in
(""|"--")
	"${_cmd}" "${_confpath}" $*
	;;
("-d")
	./libtool --mode=execute gdb --args "${_cmd}" "${_confpath}" $*
	;;
("-v")
	./libtool --mode=execute valgrind --leak-check=full --log-file='valgrind.log' "${_cmd}" "${_confpath}" $*
	;;
("-vgdb")
	./libtool --mode=execute valgrind --vgdb=yes --vgdb-error=0 --leak-check=full --log-file='valgrind.log' "${_cmd}" "${_confpath}" $*
	;;
(*)
	echo "Invalid option: ${_type}" >&2
	exit 1
	;;
esac
