ACLOCAL_AMFLAGS = -I m4
AUTOMAKE_OPTIONS = foreign subdir-objects
AM_CPPFLAGS = -Wall -Wextra -Werror -Wno-error=unused-parameter -pipe -fPIC -DPIC
AM_CXXFLAGS =

%.hpp.gch: %.hpp
	$(CXX) -x c++-header $(AM_CPPFLAGS) $(CPPFLAGS) $(AM_CXXFLAGS) $(CXXFLAGS) $< -o $@

BUILT_SOURCES = \
	src/main/precompiled.hpp.gch

CLEANFILES = \
	src/main/precompiled.hpp.gch

pkginclude_HEADERS = \
	src/main/tcp_session_base.hpp	\
	src/main/optional_map.hpp	\
	src/main/ssl_raii.hpp	\
	src/main/socket_server_base.hpp	\
	src/main/log.hpp	\
	src/main/precompiled.hpp	\
	src/main/shared_ntmbs.hpp	\
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
	src/main/file.hpp	\
	src/main/udp_server_base.hpp	\
	src/main/stream_buffer.hpp	\
	src/main/ip_port.hpp	\
	src/main/exception.hpp	\
	src/main/raii.hpp	\
	src/main/cxx_util.hpp	\
	src/main/endian.hpp	\
	src/main/csv_parser.hpp	\
	src/main/utilities.hpp	\
	src/main/atomic.hpp	\
	src/main/epoll.hpp	\
	src/main/session_base.hpp	\
	src/main/cxx_ver.hpp	\
	src/main/ssl_filter_base.hpp	\
	src/main/sock_addr.hpp	\
	src/main/virtual_shared_from_this.hpp

pkginclude_singletonsdir = $(pkgincludedir)/singletons
pkginclude_singletons_HEADERS = \
	src/main/singletons/http_servlet_depository.hpp	\
	src/main/singletons/job_dispatcher.hpp	\
	src/main/singletons/timer_daemon.hpp	\
	src/main/singletons/main_config.hpp	\
	src/main/singletons/epoll_daemon.hpp	\
	src/main/singletons/system_http_server.hpp	\
	src/main/singletons/module_depository.hpp	\
	src/main/singletons/mysql_daemon.hpp	\
	src/main/singletons/player_servlet_depository.hpp	\
	src/main/singletons/event_dispatcher.hpp	\
	src/main/singletons/profile_depository.hpp	\
	src/main/singletons/websocket_servlet_depository.hpp

pkginclude_httpdir = $(pkgincludedir)/http
pkginclude_http_HEADERS = \
	src/main/http/session.hpp	\
	src/main/http/verb.hpp	\
	src/main/http/status.hpp	\
	src/main/http/exception.hpp	\
	src/main/http/server.hpp	\
	src/main/http/utilities.hpp	\
	src/main/http/upgraded_session_base.hpp	\
	src/main/http/request.hpp

pkginclude_http_websocketdir = $(pkgincludedir)/http/websocket
pkginclude_http_websocket_HEADERS = \
	src/main/http/websocket/session.hpp	\
	src/main/http/websocket/status.hpp	\
	src/main/http/websocket/exception.hpp	\
	src/main/http/websocket/opcode.hpp

pkginclude_playerdir = $(pkgincludedir)/player
pkginclude_player_HEADERS = \
	src/main/player/client.hpp	\
	src/main/player/protocol_base.hpp	\
	src/main/player/session.hpp	\
	src/main/player/error_protocol.hpp	\
	src/main/player/protocol_generator.hpp	\
	src/main/player/status.hpp	\
	src/main/player/exception.hpp	\
	src/main/player/server.hpp

pkginclude_mysqldir = $(pkgincludedir)/mysql
pkginclude_mysql_HEADERS = \
	src/main/mysql/object_generator.hpp	\
	src/main/mysql/thread_context.hpp	\
	src/main/mysql/connection.hpp	\
	src/main/mysql/object_base.hpp	\
	src/main/mysql/callbacks.hpp	\
	src/main/mysql/exception.hpp	\
	src/main/mysql/utilities.hpp

sbin_PROGRAMS = \
	sbin/poseidon

sbin_poseidon_SOURCES = \
	src/main/main.cpp

sbin_poseidon_LDADD = \
	lib/libposeidon-main.la

lib_libposeidon_main_la_SOURCES = \
	src/main/utilities.cpp	\
	src/main/log.cpp	\
	src/main/exception.cpp	\
	src/main/profiler.cpp	\
	src/main/raii.cpp	\
	src/main/virtual_shared_from_this.cpp	\
	src/main/transaction.cpp	\
	src/main/stream_buffer.cpp	\
	src/main/shared_ntmbs.cpp	\
	src/main/optional_map.cpp	\
	src/main/csv_parser.cpp	\
	src/main/job_base.cpp	\
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
	src/main/singletons/main_config.cpp	\
	src/main/singletons/job_dispatcher.cpp	\
	src/main/singletons/mysql_daemon.cpp	\
	src/main/singletons/epoll_daemon.cpp	\
	src/main/singletons/timer_daemon.cpp	\
	src/main/singletons/player_servlet_depository.cpp	\
	src/main/singletons/http_servlet_depository.cpp	\
	src/main/singletons/websocket_servlet_depository.cpp	\
	src/main/singletons/module_depository.cpp	\
	src/main/singletons/event_dispatcher.cpp	\
	src/main/singletons/profile_depository.cpp	\
	src/main/singletons/system_http_server.cpp	\
	src/main/player/session.cpp	\
	src/main/player/server.cpp	\
	src/main/player/client.cpp	\
	src/main/player/exception.cpp	\
	src/main/http/session.cpp	\
	src/main/http/server.cpp	\
	src/main/http/status.cpp	\
	src/main/http/verb.cpp	\
	src/main/http/utilities.cpp	\
	src/main/http/upgraded_session_base.cpp	\
	src/main/http/exception.cpp	\
	src/main/http/websocket/session.cpp	\
	src/main/http/websocket/exception.cpp	\
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
