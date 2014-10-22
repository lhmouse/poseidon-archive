#include "../precompiled.hpp"
#include "../main/singletons/module_manager.hpp"
#include "../main/singletons/config_file.hpp"
#include "../main/singletons/epoll_daemon.hpp"
#include "../main/log.hpp"
#include "../main/exception.hpp"
using namespace Poseidon;

extern "C" void poseidonModuleInit(boost::weak_ptr<Module>, boost::shared_ptr<const void> &context){
	LOG_INFO("Initializing player server...");

	std::string bind("0.0.0.0");
	boost::uint16_t port = 0;
	std::string certificate;
	std::string privateKey;

	ConfigFile::get(bind, "player_bind");
	ConfigFile::get(port, "player_port");
	ConfigFile::get(certificate, "player_certificate");
	ConfigFile::get(privateKey, "player_private_key");

	context = EpollDaemon::registerPlayerServer(bind, port, certificate, privateKey);
}
