#include "../precompiled.hpp"
#include "utilities.hpp"
#include "log.hpp"
#include "exception.hpp"
#include <iostream>
using namespace Poseidon;

int main(){
	try {
		DEBUG_THROW(SystemError, ENOMEM);
	} catch(Exception &e){
		LOG_FATAL <<e.what() <<" FILE " <<e.file() <<" LINE " <<e.line();
	}
}
