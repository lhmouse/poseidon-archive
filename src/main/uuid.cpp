#include "../precompiled.hpp"
#include "uuid.hpp"
#include <boost/uuid/uuid_generators.hpp>
#include <boost/uuid/uuid_io.hpp>
#include <boost/lexical_cast.hpp>
using namespace Poseidon;

Uuid Uuid::createRandom(){
	return Uuid(boost::uuids::random_generator()());
}
Uuid Uuid::createFromString(const std::string &str){
	return Uuid(boost::lexical_cast<boost::uuids::uuid>(str));
}

Uuid::Uuid(const boost::uuids::uuid &val)
	: m_val(val)
{
}

std::string Uuid::toHex() const {
	return boost::lexical_cast<std::string>(m_val);
}
