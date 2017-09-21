#!/bin/bash

set -e

prefix="/usr/local"

pkgname="libmysqlclient-dev"
pkgversion="6.1.11"
pkglicense="GPL or Commercial (https://www.mysql.com/about/legal/licensing/oem/)"
pkggroup="https://dev.mysql.com/"
pkgsource="https://dev.mysql.com/get/Downloads/Connector-C/mysql-connector-c-${pkgversion}-src.tar.gz"
maintainer="lh_mouse"
provides="libmysqlclient-dev"

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
CFLAGS="-O3" cmake . -DCMAKE_INSTALL_PREFIX=/usr/local	\
	-DDEFAULT_CHARSET=utf8 -DDEFAULT_COLLATION=utf8_general_ci -DMYSQL_UNIX_ADDR="/var/run/mysqld/mysqld.sock"
make -j4

sudo mkdir -p "${prefix}/bin"
sudo mkdir -p "${prefix}/etc"
sudo mkdir -p "${prefix}/include"
sudo mkdir -p "${prefix}/lib"
sudo mkdir -p "${prefix}/man"
sudo mkdir -p "${prefix}/sbin"
sudo mkdir -p "${prefix}/share/doc"

sudo checkinstall --backup=no --nodoc -y --exclude=~	\
	--pkgname="${pkgname}" --pkgversion="${pkgversion}" --pkglicense="${pkglicense}" --pkggroup="${pkggroup}"	\
	--pkgsource="${pkgsource}" --maintainer="${maintainer}" --provides="${provides}"
sudo mv *.deb "${dstdir}/"
