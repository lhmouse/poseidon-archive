#!/bin/bash

set -e

prefix="/usr/local"

pkgname="libmongoc-dev"
pkgversion="1.6.3"
pkglicense="Apache"
pkggroup="http://mongoc.org/"
pkgsource="https://github.com/mongodb/mongo-c-driver/releases/download/${pkgversion}/mongo-c-driver-${pkgversion}.tar.gz"
maintainer="lh_mouse"
provides="libmongoc-1.0-dev,libbson-1.0-dev"

dstdir="$(pwd)"
tmpdir="$(pwd)/tmp"

mkdir -p "${tmpdir}"
cd "${tmpdir}"

_archive="$(basename -- ${pkgsource})"
[[ -f "${_archive}" ]] || (wget -O "${_archive}~" "${pkgsource}" && mv -f "${_archive}~" "${_archive}")
_unpackeddir="$(basename "${_archive}" ".tar.gz")"
[[ -z "${_unpackeddir}" ]] || rm -rf "${_unpackeddir}"
tar -xzvf "${_archive}"
cd "${_unpackeddir}"
CFLAGS="-O3" ./configure --prefix="${prefix}"	\
	--disable-automatic-init-and-cleanup --with-libbson=bundled
make -j4

mkdir -p "${prefix}/bin"
mkdir -p "${prefix}/etc"
mkdir -p "${prefix}/include"
mkdir -p "${prefix}/lib"
mkdir -p "${prefix}/man"
mkdir -p "${prefix}/sbin"
mkdir -p "${prefix}/share/doc"

sudo checkinstall --backup=no --nodoc -y	\
	--pkgname="${pkgname}" --pkgversion="${pkgversion}" --pkglicense="${pkglicense}" --pkggroup="${pkggroup}"	\
	--pkgsource="${pkgsource}" --maintainer="${maintainer}" --provides="${provides}"
sudo mv *.deb "${dstdir}/"
