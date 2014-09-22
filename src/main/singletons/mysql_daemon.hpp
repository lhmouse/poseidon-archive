#ifndef POSEIDON_SINGLETONS_MYSQL_DAEMON_HPP_
#define POSEIDON_SINGLETONS_MYSQL_DAEMON_HPP_

#include <boost/shared_ptr.hpp>
#include <boost/function.hpp>

namespace Poseidon {

typedef boost::function<
	void (boost::shared_ptr<class MySqlObjectBase> obj, bool result)
	> MySqlAsyncLoadCallback;

class MySqlObjectBase;

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
