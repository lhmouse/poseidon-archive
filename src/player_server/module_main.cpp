#include "../precompiled.hpp"
#include "../main/singletons/module_manager.hpp"
#include "../main/singletons/epoll_daemon.hpp"
#include "../main/config_file.hpp"
#include "../main/log.hpp"
#include "../main/exception.hpp"
using namespace Poseidon;

extern "C" void poseidonModuleInit(boost::weak_ptr<Module>, boost::shared_ptr<const void> &context){
	LOG_INFO("Initializing player server...");

	ConfigFile config("config/player_server.conf");

	std::string bind("0.0.0.0");
	boost::uint16_t port = 0;
	std::string certificate;
	std::string privateKey;

	config.get(bind, "player_server_bind");
	config.get(port, "player_server_port");
	config.get(certificate, "player_server_certificate");
	config.get(privateKey, "player_server_private_key");

	context = EpollDaemon::registerPlayerServer(bind, port, certificate, privateKey);
}
