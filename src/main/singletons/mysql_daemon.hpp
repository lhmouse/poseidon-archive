// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_SINGLETONS_MYSQL_DAEMON_HPP_
#define POSEIDON_SINGLETONS_MYSQL_DAEMON_HPP_

#include "../cxx_ver.hpp"
#include <vector>
#include <boost/shared_ptr.hpp>
#include "../mysql/callbacks.hpp"

namespace Poseidon {

namespace MySql {
	class ObjectBase;
	class Connection;
}

struct MySqlDaemon {
	struct SnapshotItem {
		unsigned thread;
		const char *table;
		unsigned long long usTotal;
	};

	static void start();
	static void stop();

	// 同步接口。
	static boost::shared_ptr<MySql::Connection> createConnection();

	// 异步接口。
	static std::vector<SnapshotItem> snapshot();

	static void waitForAllAsyncOperations();

	static void enqueueForSaving(boost::shared_ptr<const MySql::ObjectBase> object, bool toReplace, bool urgent,
		MySql::AsyncSaveCallback callback, MySql::ExceptionCallback except);
	static void enqueueForLoading(boost::shared_ptr<MySql::ObjectBase> object, std::string query,
		MySql::AsyncLoadCallback callback, MySql::ExceptionCallback except);
	static void enqueueForBatchLoading(boost::shared_ptr<MySql::ObjectBase> (*factory)(),
		const char *tableHint, std::string query,
		MySql::BatchAsyncLoadCallback callback, MySql::ExceptionCallback except);

private:
	MySqlDaemon();
};

}

#endif
