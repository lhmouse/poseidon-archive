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

const boost::weak_ptr<const char> NULL_WEAK_PTR;

}

bool SharedNtmbs::isOwning() const {
	const boost::weak_ptr<const char> ownerTest(m_ptr);
	return (ownerTest < NULL_WEAK_PTR) || (NULL_WEAK_PTR < ownerTest);
}
void SharedNtmbs::forkOwning(){
	if(!isOwning()){
		const std::size_t size = std::strlen(m_ptr.get()) + 1;
		boost::shared_ptr<char> sp(new char[size], CharArrayDeleter());
		std::memcpy(sp.get(), m_ptr.get(), size);
		m_ptr = sp;
	}
}

namespace Poseidon {

std::ostream &operator<<(std::ostream &os, const SharedNtmbs &rhs){
	return os <<rhs.get();
}

}
