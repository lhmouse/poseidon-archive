ACLOCAL_AMFLAGS = -I m4
END =

## Compiler & linker options
AM_CPPFLAGS = -pipe -pthread -Wall -Wextra -Iposeidon  \
  -D_FILE_OFFSET_BITS=64 -D_POSIX_C_SOURCE=200809 -D_GNU_SOURCE  \
  -f{strict-{aliasing,overflow},merge-all-constants,fast-math}  \
  -fno-{stack-protector,align-{functions,jumps,loops}}  \
  -fno-{devirtualize-speculatively,semantic-interposition,ident}  \
  -Werror={conversion,sign-{compare,conversion},write-strings}  \
  -Werror={return-type,double-promotion,missing-declarations}  \
  -W{missing-field-initializers,suggest-attribute=noreturn,shadow}  \
  -W{switch-enum,unused-{function,label,local-typedefs}}  \
  -Wunused-but-set-{variable,parameter}

AM_CXXFLAGS = -std=c++17 @sanitizer_flags@  \
  -fvisibility-inlines-hidden  \
  -Wno-redundant-move  \
  -Werror={non-virtual-dtor,missing-declarations}

AM_LDFLAGS = -no-undefined

AM_DEFAULT_SOURCE_EXT = .cpp
LDADD = lib/libposeidon.la
SUFFIXES = .xipp

## Initialization
noinst_LIBRARIES =
noinst_LTLIBRARIES =

nobase_include_HEADERS =
nobase_sysconf_DATA =
lib_LIBRARIES =
lib_LTLIBRARIES =
bin_PROGRAMS =

check_HEADERS =
check_LIBRARIES =
check_LTLIBRARIES =
check_PROGRAMS =

BUILT_SOURCES =
CLEANFILES =
EXTRA_DIST =
TESTS = ${check_PROGRAMS}

## Documentation
#include doc/Makefile.inc.am

## Programs and libraries
include poseidon/Makefile.inc.am

## Example add-ons
include example/Makefile.inc.am

## Tests
include test/Makefile.inc.am
