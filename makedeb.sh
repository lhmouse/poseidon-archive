#!/bin/bash

_pkgname="poseidon"
_maintainer=$(git config --get user.email)
_pkgversion=$(printf "0.%u.%s" "$(git rev-list --count HEAD)" "$(git rev-parse --short HEAD)")

sudo checkinstall --backup=no --nodoc -y --strip=no --stripso=no --pkgname="$_pkgname" --maintainer="$_maintainer" --pkgversion="$_pkgversion"
