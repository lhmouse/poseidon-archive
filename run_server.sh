#!/bin/bash

if [ "$1" == "-d" ]; then
	sudo libtool --mode=execute gdb ./sbin/poseidon
else
	sudo ./sbin/poseidon
fi
