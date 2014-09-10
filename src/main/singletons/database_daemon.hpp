#ifndef POSEIDON_SINGLETONS_DATABASE_DAEMON_HPP_
#define POSEIDON_SINGLETONS_DATABASE_DAEMON_HPP_

#include <string>
#include <boost/shared_ptr.hpp>
#include "../virtual_shared_from_this.hpp"

namespace Poseidon {

class DatabaseObjectBase
	: public virtual VirtualSharedFromThis
{
private:
	std::string m_tableName;

public:
};

struct DatabaseDaemon {
	static void start();
	static void stop();

private:
	DatabaseDaemon();
};

}

#endif
