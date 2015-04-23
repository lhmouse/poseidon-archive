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
		namespace TOKEN_CAT3(ModuleRaii_, __LINE__, _Impl_) {	\
			class Stub_ : public ::Poseidon::ModuleRaiiBase {	\
				::boost::shared_ptr<const void> init() const FINAL;	\
			} const stub_;	\
		}	\
	}	\
	::boost::shared_ptr<const void> TOKEN_CAT3(ModuleRaii_, __LINE__, _Impl_)::Stub_::init() const

#endif
