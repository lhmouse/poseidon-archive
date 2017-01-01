// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2017, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_SINGLETONS_MODULE_DEPOSITORY_HPP_
#define POSEIDON_SINGLETONS_MODULE_DEPOSITORY_HPP_

#include <string>
#include <cstddef>
#include <boost/shared_ptr.hpp>
#include "../shared_nts.hpp"

namespace Poseidon {

class Module;
class ModuleRaiiBase;

class ModuleDepository {
	friend ModuleRaiiBase;

private:
	static void register_module_raii(ModuleRaiiBase *raii, long priority);
	static void unregister_module_raii(ModuleRaiiBase *raii);

	ModuleDepository();

public:
	struct SnapshotElement {
		SharedNts real_path;
		void *base_addr;
		std::size_t ref_count;
	};

	static void start();
	static void stop();

	static boost::shared_ptr<Module> load(const char *path);
	static boost::shared_ptr<Module> load_nothrow(const char *path);
	static bool unload(const boost::shared_ptr<Module> &module);
	static bool unload(void *base_addr);

	static std::vector<SnapshotElement> snapshot();
};

}

#endif
