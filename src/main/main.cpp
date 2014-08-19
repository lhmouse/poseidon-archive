#include "../precompiled.hpp"
#include "utilities.hpp"
#include "log.hpp"
#include "exception.hpp"
#include "raii.hpp"
#include <iostream>
using namespace Poseidon;

struct foo {
	foo(){
		LOG_INFO <<"ctor of foo";
	}
	~foo(){
		LOG_INFO <<"dtor of foo";
	}
};

struct foo_deleter {
	foo *operator()() const {
		return NULL;
	}
	void operator()(foo *p) const {
		delete p;
	}
};

int main(){
	try {
		ScopedHandle<foo *, foo_deleter> h;
		h.reset(new foo);
		DEBUG_THROW(SystemError, ENOMEM);
	} catch(Exception &e){
		LOG_FATAL <<e.what() <<" FILE " <<e.file() <<" LINE " <<e.line();
	}
}
