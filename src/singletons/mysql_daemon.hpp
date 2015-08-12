// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_SINGLETONS_MYSQL_DAEMON_HPP_
#define POSEIDON_SINGLETONS_MYSQL_DAEMON_HPP_

#include "../cxx_ver.hpp"
#include <vector>
#include <boost/shared_ptr.hpp>
#include "../mysql/promise.hpp"

namespace Poseidon {

namespace MySql {
	class ObjectBase;
	class Connection;
}

struct MySqlDaemon {
	struct SnapshotElement {
		unsigned thread;
		const char *table;
		unsigned long long usTotal;
	};

	static void start();
	static void stop();

	// 同步接口。
	static boost::shared_ptr<MySql::Connection> createConnection();

	// 异步接口。
	static std::vector<SnapshotElement> snapshot();

	static void waitForAllAsyncOperations();

	// 如果返回的 exception 非空，就应当重新抛出之。
	static boost::shared_ptr<const MySql::Promise> enqueueForSaving(
		boost::shared_ptr<const MySql::ObjectBase> object, bool toReplace, bool urgent);
	static boost::shared_ptr<const MySql::Promise> enqueueForLoading(
		boost::shared_ptr<MySql::ObjectBase> object, std::string query);
	static boost::shared_ptr<const MySql::Promise> enqueueForBatchLoading(
		boost::shared_ptr<MySql::ObjectBase> (*factory)(), const char *tableHint, std::string query);

private:
	MySqlDaemon();
};

}

#endif
