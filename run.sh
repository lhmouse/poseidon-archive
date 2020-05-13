#!/bin/bash -e

if test "$1" == --help
then
  cat <<END_OF_MESSGAE
Usage:
  ./run.sh
  ./run.sh gdb --args
  ./run.sh valgrind --leak-check=full --log-file=valgrind.log
END_OF_MESSGAE
  exit 0
fi

./libtool --mode=execute -- $* ./bin/poseidon ./etc/poseidon
