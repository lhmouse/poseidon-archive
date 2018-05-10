// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_SINGLETONS_MODULE_DEPOSITORY_HPP_
#define POSEIDON_SINGLETONS_MODULE_DEPOSITORY_HPP_

#include <string>
#include <cstddef>
#include <boost/shared_ptr.hpp>
#include <boost/container/vector.hpp>
#include "../rcnts.hpp"

namespace Poseidon {

class Module_raii_base;

class Module_depository {
	friend Module_raii_base;

public:
	struct Snapshot_element {
		void *dl_handle;
		void *base_address;
		Rcnts real_path;
	};

private:
	static void register_module_raii(Module_raii_base *raii, long priority);
	static void unregister_module_raii(Module_raii_base *raii) NOEXCEPT;

	Module_depository();

public:
	static void start();
	static void stop();

	static void *load(const std::string &path);
	static void *load_nothrow(const std::string &path);
	static bool unload(void *base_address) NOEXCEPT;

	static void snapshot(boost::container::vector<Snapshot_element> &ret);
};

}

#endif
