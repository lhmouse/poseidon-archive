ACLOCAL_AMFLAGS = -I m4
AM_CPPFLAGS = -Wall -Wextra -Werror -Wconversion -Wsign-conversion -Wno-error=unused	\
	-Winvalid-pch -Wno-missing-field-initializers -Wwrite-strings -Wsuggest-attribute=noreturn	\
	-Wundef -Wshadow -Wstrict-aliasing=2 -Wstrict-overflow=2 -Wno-error=pragmas	\
	-pipe -fPIC -DPIC -pthread
AM_CXXFLAGS = -Wnon-virtual-dtor
AM_LDFLAGS = -pthread
LDADD = lib/libposeidon-main.la

%.hpp.gch: %.hpp config.h
	${AM_V_CXX}${CXX} -x c++-header ${DEFS} -I. ${AM_CPPFLAGS} ${CPPFLAGS} ${AM_CXXFLAGS} ${CXXFLAGS} -Wno-error $< -o $@

BUILT_SOURCES =	\
	poseidon/src/precompiled.hpp.gch

CLEANFILES =	\
	poseidon/src/precompiled.hpp.gch

EXTRA_DIST =	\
	etc/poseidon/main-template.conf	\
	var/poseidon/mysql_dump/placeholder	\
	var/poseidon/mongodb_dump/placeholder

pkginclude_HEADERS =	\
	poseidon/src/fwd.hpp	\
	poseidon/src/socket_base.hpp	\
	poseidon/src/tcp_session_base.hpp	\
	poseidon/src/option_map.hpp	\
	poseidon/src/ssl_raii.hpp	\
	poseidon/src/log.hpp	\
	poseidon/src/precompiled.hpp	\
	poseidon/src/rcnts.hpp	\
	poseidon/src/module_raii.hpp	\
	poseidon/src/uuid.hpp	\
	poseidon/src/tcp_server_base.hpp	\
	poseidon/src/event_base.hpp	\
	poseidon/src/vint64.hpp	\
	poseidon/src/multi_index_map.hpp	\
	poseidon/src/profiler.hpp	\
	poseidon/src/crc32.hpp	\
	poseidon/src/md5.hpp	\
	poseidon/src/sha1.hpp	\
	poseidon/src/sha256.hpp	\
	poseidon/src/hex.hpp	\
	poseidon/src/base64.hpp	\
	poseidon/src/config_file.hpp	\
	poseidon/src/tcp_client_base.hpp	\
	poseidon/src/ssl_factories.hpp	\
	poseidon/src/job_base.hpp	\
	poseidon/src/async_job.hpp	\
	poseidon/src/system_http_servlet_base.hpp	\
	poseidon/src/udp_session_base.hpp	\
	poseidon/src/udp_server_base.hpp	\
	poseidon/src/udp_client_base.hpp	\
	poseidon/src/stream_buffer.hpp	\
	poseidon/src/buffer_streams.hpp	\
	poseidon/src/ip_port.hpp	\
	poseidon/src/exception.hpp	\
	poseidon/src/tiny_exception.hpp	\
	poseidon/src/system_exception.hpp	\
	poseidon/src/raii.hpp	\
	poseidon/src/cxx_util.hpp	\
	poseidon/src/endian.hpp	\
	poseidon/src/csv_document.hpp	\
	poseidon/src/string.hpp	\
	poseidon/src/checked_arithmetic.hpp	\
	poseidon/src/time.hpp	\
	poseidon/src/errno.hpp	\
	poseidon/src/random.hpp	\
	poseidon/src/flags.hpp	\
	poseidon/src/atomic.hpp	\
	poseidon/src/session_base.hpp	\
	poseidon/src/cxx_ver.hpp	\
	poseidon/src/ssl_filter.hpp	\
	poseidon/src/sock_addr.hpp	\
	poseidon/src/virtual_shared_from_this.hpp	\
	poseidon/src/json.hpp	\
	poseidon/src/module_config.hpp	\
	poseidon/src/thread.hpp	\
	poseidon/src/mutex.hpp	\
	poseidon/src/recursive_mutex.hpp	\
	poseidon/src/condition_variable.hpp	\
	poseidon/src/promise.hpp	\
	poseidon/src/system_http_session.hpp	\
	poseidon/src/zlib.hpp

pkginclude_singletonsdir = ${pkgincludedir}/singletons
pkginclude_singletons_HEADERS =	\
	poseidon/src/singletons/job_dispatcher.hpp	\
	poseidon/src/singletons/timer_daemon.hpp	\
	poseidon/src/singletons/main_config.hpp	\
	poseidon/src/singletons/epoll_daemon.hpp	\
	poseidon/src/singletons/system_http_server.hpp	\
	poseidon/src/singletons/module_depository.hpp	\
	poseidon/src/singletons/dns_daemon.hpp	\
	poseidon/src/singletons/event_dispatcher.hpp	\
	poseidon/src/singletons/filesystem_daemon.hpp	\
	poseidon/src/singletons/profile_depository.hpp	\
	poseidon/src/singletons/workhorse_camp.hpp	\
	poseidon/src/singletons/simple_http_client_daemon.hpp
if ENABLE_MYSQL
pkginclude_singletons_HEADERS += poseidon/src/singletons/mysql_daemon.hpp
endif
if ENABLE_MONGODB
pkginclude_singletons_HEADERS += poseidon/src/singletons/mongodb_daemon.hpp
endif
if ENABLE_MAGIC
pkginclude_singletons_HEADERS += poseidon/src/singletons/magic_daemon.hpp
endif

pkginclude_httpdir = ${pkgincludedir}/http
pkginclude_http_HEADERS =	\
	poseidon/src/http/fwd.hpp	\
	poseidon/src/http/request_headers.hpp	\
	poseidon/src/http/response_headers.hpp	\
	poseidon/src/http/server_reader.hpp	\
	poseidon/src/http/server_writer.hpp	\
	poseidon/src/http/client_reader.hpp	\
	poseidon/src/http/client_writer.hpp	\
	poseidon/src/http/low_level_session.hpp	\
	poseidon/src/http/session.hpp	\
	poseidon/src/http/low_level_client.hpp	\
	poseidon/src/http/client.hpp	\
	poseidon/src/http/authentication.hpp	\
	poseidon/src/http/verbs.hpp	\
	poseidon/src/http/status_codes.hpp	\
	poseidon/src/http/exception.hpp	\
	poseidon/src/http/urlencoded.hpp	\
	poseidon/src/http/upgraded_session_base.hpp	\
	poseidon/src/http/url_param.hpp	\
	poseidon/src/http/header_option.hpp	\
	poseidon/src/http/multipart.hpp

pkginclude_websocketdir = ${pkgincludedir}/websocket
pkginclude_websocket_HEADERS =	\
	poseidon/src/websocket/fwd.hpp	\
	poseidon/src/websocket/handshake.hpp	\
	poseidon/src/websocket/reader.hpp	\
	poseidon/src/websocket/writer.hpp	\
	poseidon/src/websocket/low_level_session.hpp	\
	poseidon/src/websocket/session.hpp	\
	poseidon/src/websocket/low_level_client.hpp	\
	poseidon/src/websocket/client.hpp	\
	poseidon/src/websocket/opcodes.hpp	\
	poseidon/src/websocket/status_codes.hpp	\
	poseidon/src/websocket/exception.hpp

pkginclude_cbppdir = ${pkgincludedir}/cbpp
pkginclude_cbpp_HEADERS =	\
	poseidon/src/cbpp/fwd.hpp	\
	poseidon/src/cbpp/reader.hpp	\
	poseidon/src/cbpp/writer.hpp	\
	poseidon/src/cbpp/message_base.hpp	\
	poseidon/src/cbpp/low_level_session.hpp	\
	poseidon/src/cbpp/session.hpp	\
	poseidon/src/cbpp/low_level_client.hpp	\
	poseidon/src/cbpp/client.hpp	\
	poseidon/src/cbpp/message_generator.inl	\
	poseidon/src/cbpp/status_codes.hpp	\
	poseidon/src/cbpp/exception.hpp

if ENABLE_MYSQL
pkginclude_mysqldir = ${pkgincludedir}/mysql
pkginclude_mysql_HEADERS =	\
	poseidon/src/mysql/fwd.hpp	\
	poseidon/src/mysql/object_generator.inl	\
	poseidon/src/mysql/connection.hpp	\
	poseidon/src/mysql/object_base.hpp	\
	poseidon/src/mysql/exception.hpp	\
	poseidon/src/mysql/formatting.hpp
endif

if ENABLE_MONGODB
pkginclude_mongodbdir = ${pkgincludedir}/mongodb
pkginclude_mongodb_HEADERS =	\
	poseidon/src/mongodb/fwd.hpp	\
	poseidon/src/mongodb/object_generator.inl	\
	poseidon/src/mongodb/connection.hpp	\
	poseidon/src/mongodb/object_base.hpp	\
	poseidon/src/mongodb/exception.hpp	\
	poseidon/src/mongodb/bson_builder.hpp
endif

lib_LTLIBRARIES =	\
	lib/libposeidon-main.la

lib_libposeidon_main_la_SOURCES =	\
	poseidon/src/string.cpp	\
	poseidon/src/time.cpp	\
	poseidon/src/errno.cpp	\
	poseidon/src/random.cpp	\
	poseidon/src/log.cpp	\
	poseidon/src/exception.cpp	\
	poseidon/src/tiny_exception.cpp	\
	poseidon/src/system_exception.cpp	\
	poseidon/src/profiler.cpp	\
	poseidon/src/raii.cpp	\
	poseidon/src/virtual_shared_from_this.cpp	\
	poseidon/src/stream_buffer.cpp	\
	poseidon/src/buffer_streams.cpp	\
	poseidon/src/rcnts.cpp	\
	poseidon/src/csv_document.cpp	\
	poseidon/src/job_base.cpp	\
	poseidon/src/async_job.cpp	\
	poseidon/src/system_http_servlet_base.cpp	\
	poseidon/src/option_map.cpp	\
	poseidon/src/ssl_raii.cpp	\
	poseidon/src/ssl_filter.cpp	\
	poseidon/src/ssl_factories.cpp	\
	poseidon/src/socket_base.cpp	\
	poseidon/src/tcp_session_base.cpp	\
	poseidon/src/tcp_server_base.cpp	\
	poseidon/src/tcp_client_base.cpp	\
	poseidon/src/udp_session_base.cpp	\
	poseidon/src/udp_server_base.cpp	\
	poseidon/src/udp_client_base.cpp	\
	poseidon/src/session_base.cpp	\
	poseidon/src/event_base.cpp	\
	poseidon/src/ip_port.cpp	\
	poseidon/src/sock_addr.cpp	\
	poseidon/src/crc32.cpp	\
	poseidon/src/md5.cpp	\
	poseidon/src/sha1.cpp	\
	poseidon/src/sha256.cpp	\
	poseidon/src/hex.cpp	\
	poseidon/src/base64.cpp	\
	poseidon/src/uuid.cpp	\
	poseidon/src/config_file.cpp	\
	poseidon/src/module_raii.cpp	\
	poseidon/src/json.cpp	\
	poseidon/src/thread.cpp	\
	poseidon/src/mutex.cpp	\
	poseidon/src/recursive_mutex.cpp	\
	poseidon/src/condition_variable.cpp	\
	poseidon/src/promise.cpp	\
	poseidon/src/system_http_session.cpp	\
	poseidon/src/zlib.cpp	\
	poseidon/src/singletons/main_config.cpp	\
	poseidon/src/singletons/job_dispatcher.cpp	\
	poseidon/src/singletons/dns_daemon.cpp	\
	poseidon/src/singletons/epoll_daemon.cpp	\
	poseidon/src/singletons/timer_daemon.cpp	\
	poseidon/src/singletons/module_depository.cpp	\
	poseidon/src/singletons/event_dispatcher.cpp	\
	poseidon/src/singletons/filesystem_daemon.cpp	\
	poseidon/src/singletons/profile_depository.cpp	\
	poseidon/src/singletons/system_http_server.cpp	\
	poseidon/src/singletons/workhorse_camp.cpp	\
	poseidon/src/singletons/simple_http_client_daemon.cpp	\
	poseidon/src/cbpp/reader.cpp	\
	poseidon/src/cbpp/writer.cpp	\
	poseidon/src/cbpp/message_base.cpp	\
	poseidon/src/cbpp/low_level_session.cpp	\
	poseidon/src/cbpp/session.cpp	\
	poseidon/src/cbpp/low_level_client.cpp	\
	poseidon/src/cbpp/client.cpp	\
	poseidon/src/cbpp/exception.cpp	\
	poseidon/src/http/server_reader.cpp	\
	poseidon/src/http/server_writer.cpp	\
	poseidon/src/http/client_reader.cpp	\
	poseidon/src/http/client_writer.cpp	\
	poseidon/src/http/low_level_session.cpp	\
	poseidon/src/http/session.cpp	\
	poseidon/src/http/low_level_client.cpp	\
	poseidon/src/http/client.cpp	\
	poseidon/src/http/authentication.cpp	\
	poseidon/src/http/status_codes.cpp	\
	poseidon/src/http/verbs.cpp	\
	poseidon/src/http/urlencoded.cpp	\
	poseidon/src/http/upgraded_session_base.cpp	\
	poseidon/src/http/exception.cpp	\
	poseidon/src/http/request_headers.cpp	\
	poseidon/src/http/response_headers.cpp	\
	poseidon/src/http/url_param.cpp	\
	poseidon/src/http/header_option.cpp	\
	poseidon/src/http/multipart.cpp	\
	poseidon/src/websocket/handshake.cpp	\
	poseidon/src/websocket/reader.cpp	\
	poseidon/src/websocket/writer.cpp	\
	poseidon/src/websocket/low_level_session.cpp	\
	poseidon/src/websocket/session.cpp	\
	poseidon/src/websocket/low_level_client.cpp	\
	poseidon/src/websocket/client.cpp	\
	poseidon/src/websocket/exception.cpp
if ENABLE_MYSQL
lib_libposeidon_main_la_SOURCES +=	\
	poseidon/src/singletons/mysql_daemon.cpp	\
	poseidon/src/mysql/object_base.cpp	\
	poseidon/src/mysql/exception.cpp	\
	poseidon/src/mysql/formatting.cpp	\
	poseidon/src/mysql/connection.cpp
endif
if ENABLE_MONGODB
lib_libposeidon_main_la_SOURCES +=	\
	poseidon/src/singletons/mongodb_daemon.cpp	\
	poseidon/src/mongodb/object_base.cpp	\
	poseidon/src/mongodb/exception.cpp	\
	poseidon/src/mongodb/connection.cpp	\
	poseidon/src/mongodb/bson_builder.cpp
endif
if ENABLE_MAGIC
lib_libposeidon_main_la_SOURCES +=	\
	poseidon/src/singletons/magic_daemon.cpp
endif

lib_libposeidon_main_la_LDFLAGS =	\
	-Wl,--no-undefined -no-undefined -version-info 25:0:0

bin_PROGRAMS =	\
	bin/poseidon

bin_poseidon_SOURCES =	\
	poseidon/src/main.cpp

sysconf_DATA =

pkgsysconfdir = ${sysconfdir}/@PACKAGE@
pkgsysconf_DATA =	\
	etc/poseidon/main-template.conf

localstate_DATA =

pkglocalstatedir = ${localstatedir}/@PACKAGE@

pkglocalstatemysql_dumpdir = ${pkglocalstatedir}/mysql_dump
pkglocalstatemysql_dump_DATA =	\
	var/poseidon/mysql_dump/placeholder

pkglocalstatemongodb_dumpdir = ${pkglocalstatedir}/mongodb_dump
pkglocalstatemongodb_dump_DATA =	\
	var/poseidon/mongodb_dump/placeholder
