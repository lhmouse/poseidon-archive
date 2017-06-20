#!/bin/bash

set -e

if [[ $EUID -ne 0 ]]; then
	echo "You must run this script as root."
	exit 1
fi

prefix="/usr/local"

pkgname="libmysqlclient-dev"
pkgversion="6.1.10"
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

mkdir -p "${prefix}/bin"
mkdir -p "${prefix}/etc"
mkdir -p "${prefix}/include"
mkdir -p "${prefix}/lib"
mkdir -p "${prefix}/man"
mkdir -p "${prefix}/sbin"
mkdir -p "${prefix}/share/doc"

checkinstall --backup=no --nodoc -y	\
	--pkgname="${pkgname}" --pkgversion="${pkgversion}" --pkglicense="${pkglicense}" --pkggroup="${pkggroup}"	\
	--pkgsource="${pkgsource}" --maintainer="${maintainer}" --provides="${provides}"
mv *.deb "${dstdir}/"
