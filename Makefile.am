ACLOCAL_AMFLAGS = -I m4
AUTOMAKE_OPTIONS = foreign subdir-objects
AM_CPPFLAGS = -Wall -Wextra -Werror -Wsign-conversion -Wno-error=unused-parameter -Wno-missing-field-initializers -pipe -fPIC -DPIC
AM_CXXFLAGS =

%.hpp.gch: %.hpp
	$(CXX) -x c++-header @DEFS@ $(AM_CPPFLAGS) $(CPPFLAGS) $(AM_CXXFLAGS) $(CXXFLAGS) $< -o $@

BUILT_SOURCES = \
	src/main/precompiled.hpp.gch

CLEANFILES = \
	src/main/precompiled.hpp.gch

pkginclude_HEADERS = \
	src/main/fwd.hpp	\
	src/main/tcp_session_base.hpp	\
	src/main/optional_map.hpp	\
	src/main/ssl_raii.hpp	\
	src/main/socket_server_base.hpp	\
	src/main/log.hpp	\
	src/main/precompiled.hpp	\
	src/main/shared_nts.hpp	\
	src/main/module_raii.hpp	\
	src/main/uuid.hpp	\
	src/main/tcp_server_base.hpp	\
	src/main/event_base.hpp	\
	src/main/vint50.hpp	\
	src/main/multi_index_map.hpp	\
	src/main/profiler.hpp	\
	src/main/hash.hpp	\
	src/main/config_file.hpp	\
	src/main/tcp_client_base.hpp	\
	src/main/transaction.hpp	\
	src/main/ssl_factories.hpp	\
	src/main/job_base.hpp	\
	src/main/async_job.hpp	\
	src/main/file.hpp	\
	src/main/udp_server_base.hpp	\
	src/main/stream_buffer.hpp	\
	src/main/ip_port.hpp	\
	src/main/exception.hpp	\
	src/main/protocol_exception.hpp	\
	src/main/system_exception.hpp	\
	src/main/raii.hpp	\
	src/main/cxx_util.hpp	\
	src/main/endian.hpp	\
	src/main/csv_parser.hpp	\
	src/main/string.hpp	\
	src/main/time.hpp	\
	src/main/errno.hpp	\
	src/main/random.hpp	\
	src/main/flags.hpp	\
	src/main/atomic.hpp	\
	src/main/epoll.hpp	\
	src/main/session_base.hpp	\
	src/main/cxx_ver.hpp	\
	src/main/ssl_filter_base.hpp	\
	src/main/sock_addr.hpp	\
	src/main/virtual_shared_from_this.hpp	\
	src/main/json.hpp	\
	src/main/module_config.hpp

pkginclude_singletonsdir = $(pkgincludedir)/singletons
pkginclude_singletons_HEADERS = \
	src/main/singletons/job_dispatcher.hpp	\
	src/main/singletons/timer_daemon.hpp	\
	src/main/singletons/main_config.hpp	\
	src/main/singletons/epoll_daemon.hpp	\
	src/main/singletons/system_http_server.hpp	\
	src/main/singletons/module_depository.hpp	\
	src/main/singletons/mysql_daemon.hpp	\
	src/main/singletons/event_dispatcher.hpp	\
	src/main/singletons/profile_depository.hpp

pkginclude_httpdir = $(pkgincludedir)/http
pkginclude_http_HEADERS = \
	src/main/http/fwd.hpp	\
	src/main/http/session.hpp	\
	src/main/http/verbs.hpp	\
	src/main/http/status_codes.hpp	\
	src/main/http/exception.hpp	\
	src/main/http/utilities.hpp	\
	src/main/http/upgraded_session_base.hpp

pkginclude_websocketdir = $(pkgincludedir)/websocket
pkginclude_websocket_HEADERS = \
	src/main/websocket/fwd.hpp	\
	src/main/websocket/session.hpp	\
	src/main/websocket/opcodes.hpp	\
	src/main/websocket/status_codes.hpp	\
	src/main/websocket/exception.hpp

pkginclude_cbppdir = $(pkgincludedir)/cbpp
pkginclude_cbpp_HEADERS = \
	src/main/cbpp/fwd.hpp	\
	src/main/cbpp/client.hpp	\
	src/main/cbpp/message_base.hpp	\
	src/main/cbpp/session.hpp	\
	src/main/cbpp/error_message.hpp	\
	src/main/cbpp/message_generator.hpp	\
	src/main/cbpp/status_codes.hpp	\
	src/main/cbpp/control_codes.hpp	\
	src/main/cbpp/exception.hpp

pkginclude_mysqldir = $(pkgincludedir)/mysql
pkginclude_mysql_HEADERS = \
	src/main/mysql/fwd.hpp	\
	src/main/mysql/object_generator.hpp	\
	src/main/mysql/thread_context.hpp	\
	src/main/mysql/connection.hpp	\
	src/main/mysql/object_base.hpp	\
	src/main/mysql/callbacks.hpp	\
	src/main/mysql/exception.hpp	\
	src/main/mysql/utilities.hpp

bin_PROGRAMS = \
	bin/poseidon

bin_poseidon_SOURCES = \
	src/main/main.cpp

bin_poseidon_LDADD = \
	lib/libposeidon-main.la

lib_libposeidon_main_la_SOURCES = \
	src/main/string.cpp	\
	src/main/time.cpp	\
	src/main/errno.cpp	\
	src/main/random.cpp	\
	src/main/log.cpp	\
	src/main/exception.cpp	\
	src/main/protocol_exception.cpp	\
	src/main/system_exception.cpp	\
	src/main/profiler.cpp	\
	src/main/raii.cpp	\
	src/main/virtual_shared_from_this.cpp	\
	src/main/transaction.cpp	\
	src/main/stream_buffer.cpp	\
	src/main/shared_nts.cpp	\
	src/main/optional_map.cpp	\
	src/main/csv_parser.cpp	\
	src/main/job_base.cpp	\
	src/main/async_job.cpp	\
	src/main/ssl_raii.cpp	\
	src/main/ssl_filter_base.cpp	\
	src/main/ssl_factories.cpp	\
	src/main/tcp_session_base.cpp	\
	src/main/socket_server_base.cpp	\
	src/main/tcp_server_base.cpp	\
	src/main/tcp_client_base.cpp	\
	src/main/udp_server_base.cpp	\
	src/main/session_base.cpp	\
	src/main/epoll.cpp	\
	src/main/event_base.cpp	\
	src/main/ip_port.cpp	\
	src/main/sock_addr.cpp	\
	src/main/hash.cpp	\
	src/main/uuid.cpp	\
	src/main/file.cpp	\
	src/main/config_file.cpp	\
	src/main/module_raii.cpp	\
	src/main/json.cpp	\
	src/main/singletons/main_config.cpp	\
	src/main/singletons/job_dispatcher.cpp	\
	src/main/singletons/mysql_daemon.cpp	\
	src/main/singletons/epoll_daemon.cpp	\
	src/main/singletons/timer_daemon.cpp	\
	src/main/singletons/module_depository.cpp	\
	src/main/singletons/event_dispatcher.cpp	\
	src/main/singletons/profile_depository.cpp	\
	src/main/singletons/system_http_server.cpp	\
	src/main/cbpp/session.cpp	\
	src/main/cbpp/client.cpp	\
	src/main/cbpp/exception.cpp	\
	src/main/cbpp/message_base.cpp	\
	src/main/http/session.cpp	\
	src/main/http/status_codes.cpp	\
	src/main/http/verbs.cpp	\
	src/main/http/utilities.cpp	\
	src/main/http/upgraded_session_base.cpp	\
	src/main/http/exception.cpp	\
	src/main/websocket/session.cpp	\
	src/main/websocket/exception.cpp	\
	src/main/mysql/object_base.cpp	\
	src/main/mysql/exception.cpp	\
	src/main/mysql/utilities.cpp	\
	src/main/mysql/thread_context.cpp	\
	src/main/mysql/connection.cpp

pkgsysconfdir = $(sysconfdir)/@PACKAGE@
pkgsysconf_DATA = \
	etc/poseidon/main.conf

pkglocalstatedir = $(localstatedir)/@PACKAGE@

pkglocalstatesqldumpdir = $(pkglocalstatedir)/sqldump
pkglocalstatesqldump_DATA = \
	var/poseidon/sqldump/placeholder

lib_LTLIBRARIES = \
	lib/libposeidon-main.la	\
	lib/libposeidon-template.la

lib_libposeidon_template_la_SOURCES =	\
	src/template/mmain.cpp
