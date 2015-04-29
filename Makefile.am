ACLOCAL_AMFLAGS = -I m4
AUTOMAKE_OPTIONS = foreign subdir-objects
AM_CPPFLAGS = -Wall -Wextra -Werror -Wsign-conversion -Wno-error=unused-parameter -Wno-missing-field-initializers -pipe -fPIC -DPIC
AM_CXXFLAGS =

%.hpp.gch: %.hpp
	$(CXX) -x c++-header @DEFS@ $(AM_CPPFLAGS) $(CPPFLAGS) $(AM_CXXFLAGS) $(CXXFLAGS) $< -o $@

BUILT_SOURCES = \
	src/precompiled.hpp.gch

CLEANFILES = \
	src/precompiled.hpp.gch

pkginclude_HEADERS = \
	src/fwd.hpp	\
	src/tcp_session_base.hpp	\
	src/optional_map.hpp	\
	src/ssl_raii.hpp	\
	src/socket_server_base.hpp	\
	src/log.hpp	\
	src/precompiled.hpp	\
	src/shared_nts.hpp	\
	src/module_raii.hpp	\
	src/uuid.hpp	\
	src/tcp_server_base.hpp	\
	src/event_base.hpp	\
	src/vint50.hpp	\
	src/multi_index_map.hpp	\
	src/profiler.hpp	\
	src/hash.hpp	\
	src/config_file.hpp	\
	src/tcp_client_base.hpp	\
	src/transaction.hpp	\
	src/ssl_factories.hpp	\
	src/job_base.hpp	\
	src/async_job.hpp	\
	src/file.hpp	\
	src/udp_server_base.hpp	\
	src/stream_buffer.hpp	\
	src/ip_port.hpp	\
	src/exception.hpp	\
	src/protocol_exception.hpp	\
	src/system_exception.hpp	\
	src/raii.hpp	\
	src/cxx_util.hpp	\
	src/endian.hpp	\
	src/csv_parser.hpp	\
	src/string.hpp	\
	src/time.hpp	\
	src/errno.hpp	\
	src/random.hpp	\
	src/flags.hpp	\
	src/atomic.hpp	\
	src/epoll.hpp	\
	src/session_base.hpp	\
	src/cxx_ver.hpp	\
	src/ssl_filter_base.hpp	\
	src/sock_addr.hpp	\
	src/virtual_shared_from_this.hpp	\
	src/json.hpp	\
	src/module_config.hpp	\
	src/thread.hpp	\
	src/mutex.hpp	\
	src/condition_variable.hpp

pkginclude_singletonsdir = $(pkgincludedir)/singletons
pkginclude_singletons_HEADERS = \
	src/singletons/job_dispatcher.hpp	\
	src/singletons/timer_daemon.hpp	\
	src/singletons/main_config.hpp	\
	src/singletons/epoll_daemon.hpp	\
	src/singletons/system_http_server.hpp	\
	src/singletons/module_depository.hpp	\
	src/singletons/mysql_daemon.hpp	\
	src/singletons/event_dispatcher.hpp	\
	src/singletons/profile_depository.hpp

pkginclude_httpdir = $(pkgincludedir)/http
pkginclude_http_HEADERS = \
	src/http/fwd.hpp	\
	src/http/request_headers.hpp	\
	src/http/response_headers.hpp	\
	src/http/session.hpp	\
	src/http/client.hpp	\
	src/http/authorization.hpp	\
	src/http/verbs.hpp	\
	src/http/status_codes.hpp	\
	src/http/exception.hpp	\
	src/http/utilities.hpp	\
	src/http/chunked_writer.hpp	\
	src/http/upgraded_session_base.hpp

pkginclude_websocketdir = $(pkgincludedir)/websocket
pkginclude_websocket_HEADERS = \
	src/websocket/fwd.hpp	\
	src/websocket/session.hpp	\
	src/websocket/opcodes.hpp	\
	src/websocket/status_codes.hpp	\
	src/websocket/exception.hpp

pkginclude_cbppdir = $(pkgincludedir)/cbpp
pkginclude_cbpp_HEADERS = \
	src/cbpp/fwd.hpp	\
	src/cbpp/client.hpp	\
	src/cbpp/message_base.hpp	\
	src/cbpp/session.hpp	\
	src/cbpp/control_message.hpp	\
	src/cbpp/message_generator.hpp	\
	src/cbpp/status_codes.hpp	\
	src/cbpp/control_codes.hpp	\
	src/cbpp/exception.hpp

pkginclude_mysqldir = $(pkgincludedir)/mysql
pkginclude_mysql_HEADERS = \
	src/mysql/fwd.hpp	\
	src/mysql/object_generator.hpp	\
	src/mysql/thread_context.hpp	\
	src/mysql/connection.hpp	\
	src/mysql/object_base.hpp	\
	src/mysql/callbacks.hpp	\
	src/mysql/exception.hpp	\
	src/mysql/utilities.hpp

bin_PROGRAMS = \
	bin/poseidon

bin_poseidon_SOURCES = \
	src/main.cpp

bin_poseidon_LDADD = \
	lib/libposeidon-main.la

lib_libposeidon_main_la_SOURCES = \
	src/string.cpp	\
	src/time.cpp	\
	src/errno.cpp	\
	src/random.cpp	\
	src/log.cpp	\
	src/exception.cpp	\
	src/protocol_exception.cpp	\
	src/system_exception.cpp	\
	src/profiler.cpp	\
	src/raii.cpp	\
	src/virtual_shared_from_this.cpp	\
	src/transaction.cpp	\
	src/stream_buffer.cpp	\
	src/shared_nts.cpp	\
	src/optional_map.cpp	\
	src/csv_parser.cpp	\
	src/job_base.cpp	\
	src/async_job.cpp	\
	src/ssl_raii.cpp	\
	src/ssl_filter_base.cpp	\
	src/ssl_factories.cpp	\
	src/tcp_session_base.cpp	\
	src/socket_server_base.cpp	\
	src/tcp_server_base.cpp	\
	src/tcp_client_base.cpp	\
	src/udp_server_base.cpp	\
	src/session_base.cpp	\
	src/epoll.cpp	\
	src/event_base.cpp	\
	src/ip_port.cpp	\
	src/sock_addr.cpp	\
	src/hash.cpp	\
	src/uuid.cpp	\
	src/file.cpp	\
	src/config_file.cpp	\
	src/module_raii.cpp	\
	src/json.cpp	\
	src/thread.cpp	\
	src/mutex.cpp	\
	src/condition_variable.cpp	\
	src/singletons/main_config.cpp	\
	src/singletons/job_dispatcher.cpp	\
	src/singletons/mysql_daemon.cpp	\
	src/singletons/epoll_daemon.cpp	\
	src/singletons/timer_daemon.cpp	\
	src/singletons/module_depository.cpp	\
	src/singletons/event_dispatcher.cpp	\
	src/singletons/profile_depository.cpp	\
	src/singletons/system_http_server.cpp	\
	src/cbpp/session.cpp	\
	src/cbpp/client.cpp	\
	src/cbpp/exception.cpp	\
	src/http/session.cpp	\
	src/http/client.cpp	\
	src/http/authorization.cpp	\
	src/http/status_codes.cpp	\
	src/http/verbs.cpp	\
	src/http/utilities.cpp	\
	src/http/chunked_writer.cpp	\
	src/http/upgraded_session_base.cpp	\
	src/http/exception.cpp	\
	src/websocket/session.cpp	\
	src/websocket/exception.cpp	\
	src/mysql/object_base.cpp	\
	src/mysql/exception.cpp	\
	src/mysql/utilities.cpp	\
	src/mysql/thread_context.cpp	\
	src/mysql/connection.cpp

pkgsysconfdir = $(sysconfdir)/@PACKAGE@
pkgsysconf_DATA = \
	etc/poseidon/main-template.conf

pkglocalstatedir = $(localstatedir)/@PACKAGE@

pkglocalstatesqldumpdir = $(pkglocalstatedir)/sqldump
pkglocalstatesqldump_DATA = \
	var/poseidon/sqldump/placeholder

lib_LTLIBRARIES = \
	lib/libposeidon-main.la	\
	lib/libposeidon-template.la

lib_libposeidon_template_la_SOURCES =	\
	template-src/mmain.cpp
