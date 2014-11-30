ACLOCAL_AMFLAGS=-I m4
AUTOMAKE_OPTIONS=foreign subdir-objects
AM_CPPFLAGS=-Wall -Wextra -Werror -Wno-error=unused-parameter
AM_CXXFLAGS=

%.hpp.gch: %.hpp
	$(CXX) -x c++-header $(CPPFLAGS) $(AM_CXXFLAGS) $(CXXFLAGS) $< -o $@

BUILT_SOURCES=	\
	src/main/precompiled.hpp.gch

CLEANFILES=	\
	src/main/precompiled.hpp.gch

sbin_PROGRAMS=	\
	sbin/poseidon

sbin_poseidon_SOURCES=	\
	src/main/main.cpp

sbin_poseidon_LDADD=	\
	lib/libposeidon-main.la

lib_libposeidon_main_la_SOURCES=	\
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
	src/main/tcp_session_base.cpp	\
	src/main/tcp_session_ssl_impl.cpp	\
	src/main/socket_server_base.cpp	\
	src/main/tcp_server_base.cpp	\
	src/main/tcp_client_base.cpp	\
	src/main/udp_server_base.cpp	\
	src/main/session_base.cpp	\
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

lib_LTLIBRARIES=	\
	lib/libposeidon-main.la	\
	lib/libposeidon-player-server.la	\
	lib/libposeidon-http-server.la	\
	lib/libposeidon-template.la	\
	lib/libposeidon-game-logs.la

lib_libposeidon_player_server_la_SOURCES=	\
	src/player_server/mmain.cpp

lib_libposeidon_http_server_la_SOURCES=	\
	src/http_server/mmain.cpp

lib_libposeidon_template_la_SOURCES=	\
	src/template/mmain.cpp

lib_libposeidon_game_logs_la_SOURCES=	\
	src/game_logs/mmain.cpp
