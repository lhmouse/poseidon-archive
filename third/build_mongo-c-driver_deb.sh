#!/bin/bash

set -e

prefix="/usr/local"

pkgname="libmongoc-dev"
pkgversion="1.9.3"
pkglicense="Apache"
pkggroup="http://mongoc.org/"
pkgsource="https://github.com/mongodb/mongo-c-driver/releases/download/${pkgversion}/mongo-c-driver-${pkgversion}.tar.gz"
maintainer="lh_mouse"

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

sudo mkdir -p "${prefix}/bin"
sudo mkdir -p "${prefix}/etc"
sudo mkdir -p "${prefix}/include"
sudo mkdir -p "${prefix}/lib"
sudo mkdir -p "${prefix}/man"
sudo mkdir -p "${prefix}/sbin"
sudo mkdir -p "${prefix}/share/doc"

pushd "src/libbson"
./configure --prefix="${prefix}"
find "src/bson" -name "*.h" -execdir sed -i "s@#include <bson.h>@#include \"bson.h\"@" {} +
make -j"$(nproc)"
sudo checkinstall --backup=no --nodoc -y --addso=yes --exclude="${tmpdir}" --exclude="${HOME}"	\
	--pkgname="libbson-dev" --pkgversion="${pkgversion}" --pkglicense="${pkglicense}" --pkggroup="${pkggroup}"	\
	--pkgsource="${pkgsource}" --maintainer="${maintainer}" --provides="libbson-1.0-dev"
sudo mv *.deb "${dstdir}/"
popd

./configure --prefix="${prefix}"	\
	--disable-automatic-init-and-cleanup --with-libbson=system
find "src/mongoc" -name "*.h" -execdir sed -i "s@#include <bson.h>@#include <libbson-1.0/bson.h>@" {} +
make -j"$(nproc)"
sudo checkinstall --backup=no --nodoc -y --addso=yes --exclude="${tmpdir}" --exclude="${HOME}"	\
	--pkgname="${pkgname}" --pkgversion="${pkgversion}" --pkglicense="${pkglicense}" --pkggroup="${pkggroup}"	\
	--pkgsource="${pkgsource}" --maintainer="${maintainer}" --provides="libmongoc-1.0-dev"
sudo mv *.deb "${dstdir}/"
