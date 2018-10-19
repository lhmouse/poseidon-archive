#!/bin/bash -e

prefix="/usr/local"

pkgname="libmongoc-dev"
pkgversion="1.13.0"
pkglicense="Apache"
pkggroup="http://mongoc.org/"
pkgsource="https://github.com/mongodb/mongo-c-driver/releases/download/${pkgversion}/mongo-c-driver-${pkgversion}.tar.gz"
maintainer="lh_mouse"

dstdir="$(pwd)"
tmpdir="$(pwd)/tmp"

mkdir -p "${tmpdir}"
cd "${tmpdir}"

_archive="$(basename -- ${pkgsource})"
test -f "${_archive}" || (wget -O "${_archive}~" "${pkgsource}" && mv -f "${_archive}~" "${_archive}")
_unpackeddir="$(basename "${_archive}" ".tar.gz")"
test -z "${_unpackeddir}" || rm -rf "${_unpackeddir}"
tar -xzvf "${_archive}"
cd "${_unpackeddir}"

sudo mkdir -p "${prefix}/bin"
sudo mkdir -p "${prefix}/etc"
sudo mkdir -p "${prefix}/include"
sudo mkdir -p "${prefix}/lib"
sudo mkdir -p "${prefix}/man"
sudo mkdir -p "${prefix}/sbin"
sudo mkdir -p "${prefix}/share/doc"

echo 'install (FILES ${HEADERS} DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/bson")' >>"src/libbson/CMakeLists.txt"
echo 'install (FILES ${HEADERS} DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/mongoc")' >>"src/libmongoc/CMakeLists.txt"

cmake . -DCMAKE_INSTALL_PREFIX="${prefix}"	\
	-DENABLE_AUTOMATIC_INIT_AND_CLEANUP=OFF -DENABLE_BSON=ON
make -j"$(nproc)"
sudo checkinstall --backup=no --nodoc -y --addso=yes --exclude="${tmpdir}" --exclude="${HOME}"	\
	--pkgname="${pkgname}" --pkgversion="${pkgversion}" --pkglicense="${pkglicense}" --pkggroup="${pkggroup}"	\
	--pkgsource="${pkgsource}" --maintainer="${maintainer}" --provides="libmongoc-1.0-dev"
sudo mv *.deb "${dstdir}/"
