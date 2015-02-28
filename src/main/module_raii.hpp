// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_MODULE_RAII_HPP_
#define POSEIDON_MODULE_RAII_HPP_

#include "cxx_util.hpp"
#include <boost/shared_ptr.hpp>

namespace Poseidon {

class ModuleRaiiBase : NONCOPYABLE {
public:
	ModuleRaiiBase();
	virtual ~ModuleRaiiBase();

public:
	virtual boost::shared_ptr<const void> init() const = 0;
};

}

/*
	MODULE_RAII {
		// 写初始化代码。
	}
*/
#define MODULE_RAII	\
	namespace {	\
		namespace Impl_ {	\
			class TOKEN_CAT3(ModuleRaii_, __LINE__, _Z_) : public ::Poseidon::ModuleRaiiBase {	\
				::boost::shared_ptr<const void> init() const OVERRIDE FINAL;	\
			} const TOKEN_CAT3(ModuleRaii_, __LINE__, _z_);	\
		}	\
	}	\
	::boost::shared_ptr<const void> Impl_:: TOKEN_CAT3(ModuleRaii_, __LINE__, _Z_) ::init() const

#endif
