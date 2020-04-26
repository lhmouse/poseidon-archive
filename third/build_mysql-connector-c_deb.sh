#!/bin/bash -e

# Destination path settings
_prefix="/usr/local"
_dstdir="$(pwd)"
_version="6.1.11"
_release="1"

# Package settings
_pkgname="libmysqlclient-dev"
_pkggroup="libdevel"
_pkgsource="https://dev.mysql.com/get/Downloads/Connector-C/mysql-connector-c-${_version}-src.tar.gz"
_pkglicense="GPL v2.0"
_pkgversion="${_version}"
_pkgrelease="${_release}"
_maintainer="LH_Mouse"

# Download source tarball
_tmpdir="$(pwd)/tmp"
mkdir -p "${_tmpdir}"
cd "${_tmpdir}"

_archive="$(basename -- ${_pkgsource})"
test -f "${_archive}" || (wget -O "${_archive}~" "${_pkgsource}" && mv -f "${_archive}~" "${_archive}")

# Extract source files
_unpackeddir="$(basename "${_archive}" ".tar.gz")"
test -z "${_unpackeddir}" || rm -rf "${_unpackeddir}"
tar -xzvf "${_archive}"
cd "${_unpackeddir}"

# Ensure `checkinstall` doesn't fail
sudo mkdir -p "${_prefix}/bin"
sudo mkdir -p "${_prefix}/sbin"
sudo mkdir -p "${_prefix}/include"
sudo mkdir -p "${_prefix}/lib"
sudo mkdir -p "${_prefix}/etc"
sudo mkdir -p "${_prefix}/man"
sudo mkdir -p "${_prefix}/share/doc"

# Fuck you retards
sed -Ei '/SET\(SHARED_LIB_MINOR_VERSION/aSET(SHARED_LIB_PATCH_VERSION "0")' 'cmake/mysql_version.cmake'

# Configure and build
cmake .  \
  -DMYSQL_UNIX_ADDR="/var/run/mysqld/mysqld.sock"  \
  -DCMAKE_INSTALL_PREFIX="${_prefix}/mysql"  \
  -DINSTALL_INCLUDEDIR="../include/mysql"  \
  -DINSTALL_LIBDIR="../lib"

make -j"$(nproc)"

# Install to <prefix> and make a package
sudo "${_dstdir}/../ci/checkinstall"  \
  --backup=no --nodoc --default --strip=no --stripso=no --addso=yes  \
  --pkgname="${_pkgname}"  \
  --pkggroup="${_pkggroup}"  \
  --pkgsource="${_pkgsource}"  \
  --pkglicense="${_pkglicense}"  \
  --pkgversion="${_pkgversion}"  \
  --pkgrelease="${_pkgrelease}"  \
  --maintainer="${_maintainer}"  \
  --exclude="${_tmpdir}"  \
  --exclude="*/install_manifest.txt"

sudo mv *.deb "${_dstdir}/"
