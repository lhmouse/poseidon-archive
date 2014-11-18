// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_SINGLETONS_MYSQL_DAEMON_HPP_
#define POSEIDON_SINGLETONS_MYSQL_DAEMON_HPP_

#include "../cxx_ver.hpp"
#include <boost/shared_ptr.hpp>
#include <boost/function.hpp>

namespace Poseidon {

class MySqlObjectBase;

typedef boost::function<
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
