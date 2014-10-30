#ifndef POSEIDON_SINGLETONS_MODULE_MANAGER_HPP_
#define POSEIDON_SINGLETONS_MODULE_MANAGER_HPP_

#include <string>
#include <cstddef>
#include <boost/shared_ptr.hpp>
#include "../shared_ntmbs.hpp"

namespace Poseidon {

class Module;
class ModuleRaiiBase;

struct ModuleSnapshotItem {
	SharedNtmbs realPath;
	void *baseAddr;
	std::size_t refCount;
};

struct ModuleManager {
	static void start();
	static void stop();

	static boost::shared_ptr<Module> load(const SharedNtmbs &path);
	static boost::shared_ptr<Module> loadNoThrow(const SharedNtmbs &path);
	static bool unload(const boost::shared_ptr<Module> &module);
	static bool unload(const SharedNtmbs &realPath);
	static bool unload(void *baseAddr);

	static std::vector<ModuleSnapshotItem> snapshot();

private:
	friend class ModuleRaiiBase;

	static void registerModuleRaii(ModuleRaiiBase *raii);
	static void unregisterModuleRaii(ModuleRaiiBase *raii);

	ModuleManager();
};

}

#endif
