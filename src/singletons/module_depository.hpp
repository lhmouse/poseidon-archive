// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_SINGLETONS_MODULE_DEPOSITORY_HPP_
#define POSEIDON_SINGLETONS_MODULE_DEPOSITORY_HPP_

#include <string>
#include <cstddef>
#include <boost/shared_ptr.hpp>
#include "../shared_nts.hpp"

namespace Poseidon {

class Module;
class ModuleRaiiBase;

struct ModuleDepository {
	struct SnapshotElement {
		SharedNts realPath;
		void *baseAddr;
		std::size_t refCount;
	};

	static void start();
	static void stop();

	static boost::shared_ptr<Module> load(const char *path);
	static boost::shared_ptr<Module> loadNoThrow(const char *path);
	static bool unload(const boost::shared_ptr<Module> &module);
	static bool unload(void *baseAddr);

	static std::vector<SnapshotElement> snapshot();

private:
	friend ModuleRaiiBase;

	static void registerModuleRaii(ModuleRaiiBase *raii, long priority);
	static void unregisterModuleRaii(ModuleRaiiBase *raii);

	ModuleDepository();
};

}

#endif
