#include "../main/precompiled.hpp"
#include "../main/singletons/module_manager.hpp"
#include "../main/singletons/epoll_daemon.hpp"
#include "../main/config_file.hpp"
#include "../main/log.hpp"
#include "../main/exception.hpp"
#include <vector>
using namespace Poseidon;

extern "C" void poseidonModuleInit(std::vector<boost::shared_ptr<const void> > &contexts){
	LOG_INFO("Initializing HTTP server...");

	ConfigFile config("config/http_server.conf");

	std::string bind;
	boost::uint16_t port;
	std::string certificate;
	std::string privateKey;
	std::vector<std::string> authUserPasses;

	config.get(bind, "http_server_bind", "0.0.0.0");
	config.get(port, "http_server_port", 8860);
	config.get(certificate, "http_server_certificate", "");
	config.get(privateKey, "http_server_private_key", "");
	config.getAll(authUserPasses, "http_server_auth_user_pass");

	contexts.push_back(EpollDaemon::registerHttpServer(
		bind, port, certificate, privateKey, authUserPasses));
}
