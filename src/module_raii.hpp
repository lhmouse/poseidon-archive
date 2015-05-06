// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_MODULE_RAII_HPP_
#define POSEIDON_MODULE_RAII_HPP_

#include "cxx_util.hpp"
#include <boost/shared_ptr.hpp>
#include <stack>

namespace Poseidon {

class ModuleRaiiBase : NONCOPYABLE {
public:
	ModuleRaiiBase();
	virtual ~ModuleRaiiBase();

public:
	virtual void init(std::stack<boost::shared_ptr<const void> > &raiiHandles) const = 0;
};

}

/*
	MODULE_RAII(handles){
		AUTO(foo, boost::make_shared<Foo>());
		handles.push(STD_MOVE_IDN(foo));
	}
*/
#define MODULE_RAII(handles_)	\
	namespace {	\
		namespace TOKEN_CAT3(ModuleRaii_, __LINE__, _Impl_) {	\
			class Stub_ : public ::Poseidon::ModuleRaiiBase {	\
				void init(std::stack<boost::shared_ptr<const void> > &) const FINAL;	\
			} const stub_;	\
		}	\
	}	\
	void TOKEN_CAT3(ModuleRaii_, __LINE__, _Impl_)::Stub_::init(	\
		std::stack<boost::shared_ptr<const void> > & (handles_) ) const

#endif
