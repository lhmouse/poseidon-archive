#!/bin/bash

set -e

pkgname="mongo-c-driver"
pkgversion="1.4.0"
pkglicense="Apache"
pkggroup="http://mongoc.org/"
pkgsource="https://github.com/mongodb/mongo-c-driver/releases/download/${pkgversion}/mongo-c-driver-${pkgversion}.tar.gz"
maintainer="lh_mouse"
provides="mongo-c-driver,bson"

dstdir="$(pwd)"
tempdir="${dstdir}/temp-$PPID-$SECONDS-$RANDOM"

if [[ $EUID -ne 0 ]]; then
	echo "You must run this script as root."
	exit 1
fi

[[ -z "${tempdir}" ]] || rm -rf "${tempdir}"
mkdir -p "${tempdir}"
cd "${tempdir}"

_archive="$(basename -- ${pkgsource})"
wget -O "${_archive}" "${pkgsource}"
tar -xzvf "${_archive}"

cd "$(basename "${_archive}" ".tar.gz")"
CFLAGS='-O3' ./configure --disable-automatic-init-and-cleanup --prefix=/usr
make -j4
checkinstall -y --pkgname="${pkgname}" --pkgversion="${pkgversion}" --pkglicense="${pkglicense}" --pkggroup="${pkggroup}"	\
	--pkgsource="${pkgsource}" --maintainer="${maintainer}" --provides="${provides}"
mv *.deb "${dstdir}"/

[[ -z "${tempdir}" ]] || rm -rf "${tempdir}"
