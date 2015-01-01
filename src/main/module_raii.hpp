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
	virtual boost::shared_ptr<void> init() const = 0;
};

}

#define MODULE_RAII_BEGIN	\
	namespace {	\
		class UNIQUE_ID : public ::Poseidon::ModuleRaiiBase {	\
			::boost::shared_ptr<void> init() const OVERRIDE FINAL {

#define MODULE_RAII_END	\
			}	\
		} const TOKEN_CAT2(UNIQUE_ID, mraii_);	\
	}

#define MODULE_RAII(...)	\
	MODULE_RAII_BEGIN __VA_ARGS__ MODULE_RAII_END

#endif
