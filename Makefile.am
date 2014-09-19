PRECOMPILED_HEADER=src/precompiled.hpp

ACLOCAL_AMFLAGS=-I m4
AUTOMAKE_OPTIONS=foreign nostdinc subdir-objects
AM_CPPFLAGS=-Wall -Wextra -Werror	\
	-Wsuggest-attribute=noreturn -include $(PRECOMPILED_HEADER)
AM_CXXFLAGS=

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
	src/main/profiler.cpp	\
	src/main/raii.cpp	\
	src/main/virtual_shared_from_this.cpp	\
	src/main/transaction.cpp	\
	src/main/stream_buffer.cpp	\
	src/main/optional_map.cpp	\
	src/main/singletons/config_file.cpp	\
	src/main/singletons/job_dispatcher.cpp	\
	src/main/singletons/mysql_daemon.cpp	\
	src/main/singletons/epoll_daemon.cpp	\
	src/main/singletons/timer_manager_daemon.cpp	\
	src/main/singletons/player_servlet_manager.cpp	\
	src/main/singletons/http_servlet_manager.cpp	\
	src/main/http/session.cpp	\
	src/main/http/status.cpp	\
	src/main/http/utilities.cpp	\
	src/main/mysql/field.cpp	\
	src/main/mysql/object_base.cpp	\
	src/main/job_base.cpp	\
	src/main/tcp_session_base.cpp	\
	src/main/player/session.cpp	\
	src/main/socket_server.cpp	\
	src/main/tcp_client_base.cpp

lib_libposeidon_template_la_SOURCES=	\
	src/template/module_main.cpp

lib_libposeidon_template_la_LIBADD=	\
	lib/libposeidon-main.la
