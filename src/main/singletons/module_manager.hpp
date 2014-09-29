#ifndef POSEIDON_SINGLETONS_MODULE_MANAGER_HPP_
#define POSEIDON_SINGLETONS_MODULE_MANAGER_HPP_

#include <string>
#include <vector>
#include <cstddef>
#include <boost/shared_ptr.hpp>

namespace Poseidon {

struct ModuleInfo {
	std::string name;
	std::size_t refCount;
};

class Module;

struct ModuleManager {
	static void start();
	static void stop();

	static boost::shared_ptr<Module> get(const std::string &path);
	static std::vector<ModuleInfo> getLoadedList();

	static boost::shared_ptr<Module> load(const std::string &path);
	static bool unload(const std::string &path);

private:
	ModuleManager();
};

}

#endif
