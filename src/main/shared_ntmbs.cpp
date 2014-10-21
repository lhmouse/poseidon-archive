#include "../cxx_ver.hpp"
#include "shared_ntmbs.hpp"
using namespace Poseidon;

namespace {

struct CharArrayDeleter {
	void operator()(char *ptr) const {
		delete[] ptr;
	}
};

}

SharedNtmbs SharedNtmbs::createOwning(const char *str, std::size_t len){
	boost::shared_ptr<char> sp(new char[len + 1], CharArrayDeleter());
	std::memcpy(sp.get(), str, len);
	sp.get()[len] = 0;
	return SharedNtmbs(sp);
}
