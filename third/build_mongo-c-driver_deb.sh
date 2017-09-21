#!/bin/bash

set -e

prefix="/usr/local"

pkgname="libmongoc-dev"
pkgversion="1.8.0"
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

sudo mkdir -p "${prefix}/bin"
sudo mkdir -p "${prefix}/etc"
sudo mkdir -p "${prefix}/include"
sudo mkdir -p "${prefix}/lib"
sudo mkdir -p "${prefix}/man"
sudo mkdir -p "${prefix}/sbin"
sudo mkdir -p "${prefix}/share/doc"

sudo checkinstall --backup=no --nodoc -y --exclude=$(readlink -f ~)	\
	--pkgname="${pkgname}" --pkgversion="${pkgversion}" --pkglicense="${pkglicense}" --pkggroup="${pkggroup}"	\
	--pkgsource="${pkgsource}" --maintainer="${maintainer}" --provides="${provides}"
sudo mv *.deb "${dstdir}/"
