ACLOCAL_AMFLAGS = -I m4
AUTOMAKE_OPTIONS = foreign subdir-objects
AM_CPPFLAGS = -Wall -Wextra -Werror -Wsign-conversion -Wno-error=unused-parameter -Winvalid-pch -Wno-missing-field-initializers	\
	-Wwrite-strings -Wsuggest-attribute=noreturn -Wundef -Wshadow -Wstrict-aliasing=2 -Wstrict-overflow=2	\
	-pipe -fPIC -DPIC -pthread $(openssl_CFLAGS) $(bson_CFLAGS) $(mongoc_CFLAGS) $(zlib_CFLAGS)
AM_CXXFLAGS =
AM_LDFLAGS = $(openssl_LIBS) $(bson_LIBS) $(mongoc_LIBS) $(zlib_LIBS)

%.hpp.gch: %.hpp
	$(CXX) -x c++-header @DEFS@ $(AM_CPPFLAGS) $(CPPFLAGS) $(AM_CXXFLAGS) $(CXXFLAGS) $< -o $@

BUILT_SOURCES = \
	src/precompiled.hpp.gch

CLEANFILES = \
	src/precompiled.hpp.gch

EXTRA_DIST = \
	etc/poseidon/main-template.conf	\
	var/poseidon/mysql_dump/placeholder	\
	var/poseidon/mongodb_dump/placeholder

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
	src/vint64.hpp	\
	src/multi_index_map.hpp	\
	src/profiler.hpp	\
	src/crc32.hpp	\
	src/md5.hpp	\
	src/sha1.hpp	\
	src/sha256.hpp	\
	src/hex.hpp	\
	src/base64.hpp	\
	src/config_file.hpp	\
	src/tcp_client_base.hpp	\
	src/ssl_factories.hpp	\
	src/job_base.hpp	\
	src/async_job.hpp	\
	src/udp_server_base.hpp	\
	src/stream_buffer.hpp	\
	src/buffer_streams.hpp	\
	src/ip_port.hpp	\
	src/exception.hpp	\
	src/protocol_exception.hpp	\
	src/system_exception.hpp	\
	src/raii.hpp	\
	src/cxx_util.hpp	\
	src/endian.hpp	\
	src/csv_document.hpp	\
	src/string.hpp	\
	src/checked_arithmetic.hpp	\
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
	src/recursive_mutex.hpp	\
	src/condition_variable.hpp	\
	src/job_promise.hpp	\
	src/zlib.hpp

pkginclude_singletonsdir = $(pkgincludedir)/singletons
pkginclude_singletons_HEADERS = \
	src/singletons/job_dispatcher.hpp	\
	src/singletons/timer_daemon.hpp	\
	src/singletons/main_config.hpp	\
	src/singletons/epoll_daemon.hpp	\
	src/singletons/system_http_server.hpp	\
	src/singletons/module_depository.hpp	\
	src/singletons/mysql_daemon.hpp	\
	src/singletons/mongodb_daemon.hpp	\
	src/singletons/dns_daemon.hpp	\
	src/singletons/event_dispatcher.hpp	\
	src/singletons/filesystem_daemon.hpp	\
	src/singletons/profile_depository.hpp

pkginclude_httpdir = $(pkgincludedir)/http
pkginclude_http_HEADERS = \
	src/http/fwd.hpp	\
	src/http/request_headers.hpp	\
	src/http/response_headers.hpp	\
	src/http/server_reader.hpp	\
	src/http/server_writer.hpp	\
	src/http/client_reader.hpp	\
	src/http/client_writer.hpp	\
	src/http/low_level_session.hpp	\
	src/http/session.hpp	\
	src/http/low_level_client.hpp	\
	src/http/client.hpp	\
	src/http/authorization.hpp	\
	src/http/verbs.hpp	\
	src/http/status_codes.hpp	\
	src/http/exception.hpp	\
	src/http/urlencoded.hpp	\
	src/http/upgraded_session_base.hpp	\
	src/http/upgraded_client_base.hpp	\
	src/http/url_param.hpp	\
	src/http/header_option.hpp	\
	src/http/multipart.hpp

pkginclude_websocketdir = $(pkgincludedir)/websocket
pkginclude_websocket_HEADERS = \
	src/websocket/fwd.hpp	\
	src/websocket/handshake.hpp	\
	src/websocket/reader.hpp	\
	src/websocket/writer.hpp	\
	src/websocket/low_level_session.hpp	\
	src/websocket/session.hpp	\
	src/websocket/low_level_client.hpp	\
	src/websocket/client.hpp	\
	src/websocket/opcodes.hpp	\
	src/websocket/status_codes.hpp	\
	src/websocket/exception.hpp

pkginclude_cbppdir = $(pkgincludedir)/cbpp
pkginclude_cbpp_HEADERS = \
	src/cbpp/fwd.hpp	\
	src/cbpp/reader.hpp	\
	src/cbpp/writer.hpp	\
	src/cbpp/message_base.hpp	\
	src/cbpp/low_level_session.hpp	\
	src/cbpp/session.hpp	\
	src/cbpp/low_level_client.hpp	\
	src/cbpp/client.hpp	\
	src/cbpp/message_generator.hpp	\
	src/cbpp/status_codes.hpp	\
	src/cbpp/exception.hpp

pkginclude_mysqldir = $(pkgincludedir)/mysql
pkginclude_mysql_HEADERS = \
	src/mysql/fwd.hpp	\
	src/mysql/object_generator.hpp	\
	src/mysql/thread_context.hpp	\
	src/mysql/connection.hpp	\
	src/mysql/object_base.hpp	\
	src/mysql/exception.hpp	\
	src/mysql/formatting.hpp

pkginclude_mongodbdir = $(pkgincludedir)/mongodb
pkginclude_mongodb_HEADERS = \
	src/mongodb/fwd.hpp	\
	src/mongodb/object_generator.hpp	\
	src/mongodb/connection.hpp	\
	src/mongodb/object_base.hpp	\
	src/mongodb/exception.hpp	\
	src/mongodb/bson_builder.hpp

bin_PROGRAMS = \
	bin/poseidon

bin_poseidon_SOURCES = \
	src/main.cpp

bin_poseidon_LDADD = \
	lib/libposeidon-main.la	\
	$(openssl_LIBS)

lib_LTLIBRARIES = \
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
	src/stream_buffer.cpp	\
	src/buffer_streams.cpp	\
	src/shared_nts.cpp	\
	src/csv_document.cpp	\
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
	src/crc32.cpp	\
	src/md5.cpp	\
	src/sha1.cpp	\
	src/sha256.cpp	\
	src/hex.cpp	\
	src/base64.cpp	\
	src/uuid.cpp	\
	src/config_file.cpp	\
	src/module_raii.cpp	\
	src/json.cpp	\
	src/thread.cpp	\
	src/mutex.cpp	\
	src/recursive_mutex.cpp	\
	src/condition_variable.cpp	\
	src/job_promise.cpp	\
	src/zlib.cpp	\
	src/singletons/main_config.cpp	\
	src/singletons/job_dispatcher.cpp	\
	src/singletons/mysql_daemon.cpp	\
	src/singletons/mongodb_daemon.cpp	\
	src/singletons/dns_daemon.cpp	\
	src/singletons/epoll_daemon.cpp	\
	src/singletons/timer_daemon.cpp	\
	src/singletons/module_depository.cpp	\
	src/singletons/event_dispatcher.cpp	\
	src/singletons/filesystem_daemon.cpp	\
	src/singletons/profile_depository.cpp	\
	src/singletons/system_http_server.cpp	\
	src/cbpp/reader.cpp	\
	src/cbpp/writer.cpp	\
	src/cbpp/message_base.cpp	\
	src/cbpp/low_level_session.cpp	\
	src/cbpp/session.cpp	\
	src/cbpp/low_level_client.cpp	\
	src/cbpp/client.cpp	\
	src/cbpp/exception.cpp	\
	src/http/server_reader.cpp	\
	src/http/server_writer.cpp	\
	src/http/client_reader.cpp	\
	src/http/client_writer.cpp	\
	src/http/low_level_session.cpp	\
	src/http/session.cpp	\
	src/http/low_level_client.cpp	\
	src/http/client.cpp	\
	src/http/authorization.cpp	\
	src/http/status_codes.cpp	\
	src/http/verbs.cpp	\
	src/http/urlencoded.cpp	\
	src/http/upgraded_session_base.cpp	\
	src/http/upgraded_client_base.cpp	\
	src/http/exception.cpp	\
	src/http/request_headers.cpp	\
	src/http/response_headers.cpp	\
	src/http/url_param.cpp	\
	src/http/header_option.cpp	\
	src/http/multipart.cpp	\
	src/websocket/handshake.cpp	\
	src/websocket/reader.cpp	\
	src/websocket/writer.cpp	\
	src/websocket/low_level_session.cpp	\
	src/websocket/session.cpp	\
	src/websocket/low_level_client.cpp	\
	src/websocket/client.cpp	\
	src/websocket/exception.cpp	\
	src/mysql/object_base.cpp	\
	src/mysql/exception.cpp	\
	src/mysql/formatting.cpp	\
	src/mysql/thread_context.cpp	\
	src/mysql/connection.cpp	\
	src/mongodb/object_base.cpp	\
	src/mongodb/exception.cpp	\
	src/mongodb/connection.cpp	\
	src/mongodb/bson_builder.cpp

pkgsysconfdir = $(sysconfdir)/@PACKAGE@
pkgsysconf_DATA = \
	etc/poseidon/main-template.conf

pkglocalstatedir = $(localstatedir)/@PACKAGE@

pkglocalstatemysql_dumpdir = $(pkglocalstatedir)/mysql_dump
pkglocalstatemysql_dump_DATA = \
	var/poseidon/mysql_dump/placeholder

pkglocalstatemongodb_dumpdir = $(pkglocalstatedir)/mongodb_dump
pkglocalstatemongodb_dump_DATA = \
	var/poseidon/mongodb_dump/placeholder
