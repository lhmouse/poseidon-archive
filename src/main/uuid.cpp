#include "../precompiled.hpp"
#include "uuid.hpp"
#include <boost/thread/mutex.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/uuid/uuid_generators.hpp>
#include <boost/uuid/uuid_io.hpp>
#include <boost/lexical_cast.hpp>
#include "utilities.hpp"
using namespace Poseidon;

namespace {

boost::mutex g_mutex;
boost::random::mt19937 g_rng(getMonoClock());
boost::uuids::random_generator g_generator(g_rng);

}

Uuid Uuid::createRandom(){
	const boost::mutex::scoped_lock lock(g_mutex);
	return Uuid(g_generator());
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
