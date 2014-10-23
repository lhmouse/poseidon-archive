#ifndef POSEIDON_UUID_HPP_
#define POSEIDON_UUID_HPP_

#include <boost/uuid/uuid.hpp>

namespace Poseidon {

class Uuid {
public:
	static Uuid createRandom();
	static Uuid createFromString(const std::string &str);

private:
	boost::uuids::uuid m_val;

private:
	explicit Uuid(const boost::uuids::uuid &val);

public:
	std::string toHex() const;
};

}

#endif
