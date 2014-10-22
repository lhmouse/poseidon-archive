#ifndef POSEIDON_SINGLETONS_MODULE_MANAGER_HPP_
#define POSEIDON_SINGLETONS_MODULE_MANAGER_HPP_

#include <string>
#include <vector>
#include <cstddef>
#include <boost/shared_ptr.hpp>

namespace Poseidon {

class Module;

extern "C" void poseidonModuleInit(
	boost::weak_ptr<Module> module, boost::shared_ptr<const void> &context);

struct ModuleInfo {
	std::string name;
	std::size_t refCount;
};

struct ModuleManager {
	static void start();
	static void stop();

	static boost::shared_ptr<Module> get(const std::string &path);
	static std::vector<ModuleInfo> getLoadedList();

	static boost::shared_ptr<Module> load(const std::string &path);
	static boost::shared_ptr<Module> loadNoThrow(const std::string &path);
	static bool unload(const std::string &path);

private:
	ModuleManager();
};

}

#endif
