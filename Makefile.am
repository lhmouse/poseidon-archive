PRECOMPILED_HEADER=src/precompiled.hpp

ACLOCAL_AMFLAGS=-I m4
AUTOMAKE_OPTIONS=foreign nostdinc subdir-objects
AM_CPPFLAGS=-include $(PRECOMPILED_HEADER)
AM_CXXFLAGS=-Wall -Werror

BUILT_SOURCES=	\
	$(PRECOMPILED_HEADER).gch
	
CLEANFILES=	\
	$(PRECOMPILED_HEADER).gch
		
%.hpp.gch: %.hpp
	$(CXX) -x c++-header $(CPPFLAGS) $(AM_CXXFLAGS) $(CXXFLAGS) $< -o $@

sbin_PROGRAMS=	\
	sbin/poseidon

sbin_poseidon_SOURCES=	\
	src/main/main.cpp

sbin_poseidon_LDADD=	\
	lib/libposeidon-main.la

lib_LTLIBRARIES=	\
	lib/libposeidon-main.la	\
	lib/libposeidon-template.la

lib_libposeidon_main_la_SOURCES=	\
	src/main/utilities.cpp	\
	src/main/log.cpp	\
	src/main/job_dispatcher.cpp

lib_libposeidon_template_la_SOURCES=	\
	#src/template/module_main.cpp

lib_libposeidon_template_la_LIBADD=	\
	lib/libposeidon-main.la
