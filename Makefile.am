ACLOCAL_AMFLAGS = -I m4
END =

## Compiler options
AM_CPPFLAGS = -pipe -pthread -Wall -Wextra -Isrc -U_FORTIFY_SOURCE  \
  -D_FILE_OFFSET_BITS=64 -D_POSIX_C_SOURCE=200809 -D_GNU_SOURCE  \
  -f{strict-{aliasing,overflow},merge-all-constants,fast-math}  \
  -fno-{devirtualize-speculatively,semantic-interposition,ident}  \
  -Werror={conversion,sign-{compare,conversion},write-strings,return-type}  \
  -Werror={double-promotion,aggressive-loop-optimizations}  \
  -W{switch-enum,suggest-attribute=noreturn,shadow,missing-field-initializers}  \
  -Wunused-{function,label,local-typedefs,{,but-set-}{variable,parameter}}

AM_CXXFLAGS = -std=c++17 -fno-gnu-keywords -fvisibility-inlines-hidden  \
  -Wno-redundant-move -Werror={non-virtual-dtor,missing-declarations}

if enable_address_sanitizer
AM_CXXFLAGS += -fsanitize=address
endif

if enable_thread_sanitizer
AM_CXXFLAGS += -fsanitize=thread
endif

AM_DEFAULT_SOURCE_EXT = .cpp
SUFFIXES = .xipp .xipp.gch
LDADD = lib/libposeidon.la

## Initialization
noinst_LIBRARIES =
noinst_LTLIBRARIES =

include_HEADERS =
lib_LIBRARIES =
lib_LTLIBRARIES =
bin_PROGRAMS =
sysconf_DATA =

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
include src/Makefile.inc.am

if enable_mysql
include src/mysql/Makefile.inc.am
endif

if enable_mongodb
include src/mongodb/Makefile.inc.am
endif

## Sample add-ons
include addon/Makefile.inc.am

## Configuration files
include etc/poseidon/Makefile.inc.am

## Tests
#include test/Makefile.inc.am
