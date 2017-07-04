// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2017, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_SINGLETONS_MODULE_DEPOSITORY_HPP_
#define POSEIDON_SINGLETONS_MODULE_DEPOSITORY_HPP_

#include <string>
#include <cstddef>
#include <boost/shared_ptr.hpp>
#include "../shared_nts.hpp"

namespace Poseidon {

class ModuleRaiiBase;

class ModuleDepository {
	friend ModuleRaiiBase;

private:
	static void register_module_raii(ModuleRaiiBase *raii, long priority);
	static void unregister_module_raii(ModuleRaiiBase *raii) NOEXCEPT;

	ModuleDepository();

public:
	struct SnapshotElement {
		void *dl_handle;
		void *base_address;
		SharedNts real_path;
	};

	static void start();
	static void stop();

	static void *load(const std::string &path);
	static void *load_nothrow(const std::string &path);
	static bool unload(void *base_address) NOEXCEPT;

	static std::vector<SnapshotElement> snapshot();
};

}

#endif
