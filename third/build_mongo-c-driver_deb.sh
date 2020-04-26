#!/bin/bash -e

# Destination path settings
_prefix="/usr/local"
_dstdir="$(pwd)"
_version="1.16.2"
_release="1"

# Package settings
_pkgname="libmongoc-dev"
_pkggroup="libdevel"
_pkgsource="https://github.com/mongodb/mongo-c-driver/releases/download/${_version}/mongo-c-driver-${_version}.tar.gz"
_pkglicense="Apache 2.0"
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

# Replace stupid header paths with sane ones
echo 'install (FILES ${HEADERS} DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/bson")' >> "src/libbson/CMakeLists.txt"
echo 'install (FILES ${HEADERS} DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/mongoc")' >> "src/libmongoc/CMakeLists.txt"

# Configure and build
cmake .  \
  -DENABLE_AUTOMATIC_INIT_AND_CLEANUP=OFF  \
  -DENABLE_BSON=ON  \
  -DCMAKE_INSTALL_PREFIX="${_prefix}"

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
