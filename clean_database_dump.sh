#!/bin/bash

_var="$(pwd)/var/poseidon"

rm -vf "${_var}/mysql_dump/*.log"
rm -vf "${_var}/mongodb_dump/*.log"
