#include "cxx_ver.hpp"
#include "shared_ntmbs.hpp"
#include <iostream>
#include <boost/weak_ptr.hpp>
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

bool SharedNtmbs::isOwning() const {
	const boost::weak_ptr<const char> tmp(m_ptr);
	const boost::weak_ptr<const char> null;
	return (tmp < null) || (null < tmp);
}

namespace Poseidon {

std::ostream &operator<<(std::ostream &os, const SharedNtmbs &rhs){
	return os <<rhs.get();
}

}
