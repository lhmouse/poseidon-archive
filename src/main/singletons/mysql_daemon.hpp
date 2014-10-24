#ifndef POSEIDON_SINGLETONS_MYSQL_DAEMON_HPP_
#define POSEIDON_SINGLETONS_MYSQL_DAEMON_HPP_

#include "../cxx_ver.hpp"
#include <boost/shared_ptr.hpp>

#ifdef POSEIDON_CXX11
#   include <functional>
#else
#   include <tr1/functional>
#endif

namespace Poseidon {

class MySqlObjectBase;

typedef TR1::function<
	void (boost::shared_ptr<MySqlObjectBase> obj)
	> MySqlAsyncLoadCallback;

struct MySqlDaemon {
	static void start();
	static void stop();

	static void waitForAllAsyncOperations();

	static void pendForSaving(boost::shared_ptr<const MySqlObjectBase> object);
	static void pendForLoading(boost::shared_ptr<MySqlObjectBase> object,
		std::string filter, MySqlAsyncLoadCallback callback);

private:
	MySqlDaemon();
};

}

#endif
