#include "../precompiled.hpp"
#include "../main/singletons/module_manager.hpp"
#include "../main/singletons/config_file.hpp"
#include "../main/singletons/epoll_daemon.hpp"
#include "../main/log.hpp"
#include "../main/exception.hpp"
#include <vector>
using namespace Poseidon;

extern "C" void poseidonModuleInit(boost::weak_ptr<Module>, boost::shared_ptr<const void> &context){
	LOG_INFO("Initializing HTTP server...");

	std::string bind("0.0.0.0");
	boost::uint16_t port = 0;
	std::string certificate;
	std::string privateKey;
	std::vector<std::string> authUserPasses;

	ConfigFile::get(bind, "http_server_bind");
	ConfigFile::get(port, "http_server_port");
	ConfigFile::get(certificate, "http_server_certificate");
	ConfigFile::get(privateKey, "http_server_private_key");
	ConfigFile::getAll(authUserPasses, "http_server_auth_user_pass");

	context = EpollDaemon::registerHttpServer(bind, port, certificate, privateKey, authUserPasses);
}
