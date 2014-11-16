#include "precompiled.hpp"
#include "cxx_ver.hpp"
#include "ip_port.hpp"
#include <ostream>
using namespace Poseidon;

namespace Poseidon {

std::ostream &operator<<(std::ostream &os, const IpPort &rhs){
	return os <<rhs.ip <<':' <<rhs.port;
}
std::wostream &operator<<(std::wostream &os, const IpPort &rhs){
	return os <<rhs.ip <<':' <<rhs.port;
}

}
