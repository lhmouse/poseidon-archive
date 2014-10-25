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
	SharedNtmbs realPath;
	std::size_t refCount;
};

struct ModuleManager {
	static void start();
	static void stop();

	static boost::shared_ptr<Module> get(const SharedNtmbs &realPath);
	static boost::shared_ptr<Module> assertCurrent() __attribute__((__noinline__));
	static boost::shared_ptr<Module> load(const SharedNtmbs &path);
	static boost::shared_ptr<Module> loadNoThrow(const SharedNtmbs &path);
	static bool unload(const boost::shared_ptr<Module> &module);
	static bool unload(const SharedNtmbs &path);

	static ModuleSnapshotItem snapshot(const boost::shared_ptr<Module> &module);
	static std::vector<ModuleSnapshotItem> snapshot();

private:
	ModuleManager();
};

typedef boost::weak_ptr<Module> WeakModule;
typedef std::vector<boost::shared_ptr<const void> > ModuleContexts;

}

extern "C" void poseidonModuleInit(Poseidon::WeakModule module, Poseidon::ModuleContexts &contexts);

#endif
