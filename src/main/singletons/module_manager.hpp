#ifndef POSEIDON_SINGLETONS_MODULE_MANAGER_HPP_
#define POSEIDON_SINGLETONS_MODULE_MANAGER_HPP_

#include <string>
#include <vector>
#include <cstddef>
#include <boost/shared_ptr.hpp>
#include "../shared_ntmbs.hpp"

namespace Poseidon {

class Module;

struct ModuleSnapshotItem {
	SharedNtmbs path;
	std::size_t refCount;
};

struct ModuleManager {
	static void start();
	static void stop();

	static boost::shared_ptr<Module> get(const std::string &path);
	static boost::shared_ptr<Module> load(const std::string &path);
	static boost::shared_ptr<Module> loadNoThrow(const std::string &path);
	static bool unload(const std::string &path);

	static std::vector<ModuleSnapshotItem> snapshot();

private:
	ModuleManager();
};

typedef boost::weak_ptr<Module> WeakModule;
typedef std::vector<boost::shared_ptr<const void> > ModuleContexts;

extern "C" void poseidonModuleInit(WeakModule module, ModuleContexts &contexts);

}

#endif
