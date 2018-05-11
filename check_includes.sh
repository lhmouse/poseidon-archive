#!/bin/bash

set -e

for _file in $(find -L "src" -name "*.hpp"); do
	_cmd="gcc -std=c++98 -x c++ -c -o /dev/null  ${_file}"
	echo "Checking \`#include\` directives:  ${_cmd}"
	${_cmd}
done
