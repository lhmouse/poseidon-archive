#include "../main/precompiled.hpp"
#include "../main/singletons/epoll_daemon.hpp"
#include "../main/config_file.hpp"
#include "../main/log.hpp"
#include "../main/exception.hpp"
#include "../main/module_raii.hpp"
using namespace Poseidon;

MODULE_RAII(
	LOG_POSEIDON_INFO("Initializing player server...");

	ConfigFile config("config/player_server.conf");
	return EpollDaemon::registerPlayerServer(
		config.get<std::size_t>("player_server_category", 1),
		config.get<std::string>("player_server_bind", "0.0.0.0"),
		config.get<boost::uint16_t>("player_server_port", 8850),
		config.get<std::string>("player_server_certificate", ""),
		config.get<std::string>("player_server_private_key", "")
	);
)
