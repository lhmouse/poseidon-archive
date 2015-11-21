// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_SINGLETONS_MYSQL_DAEMON_HPP_
#define POSEIDON_SINGLETONS_MYSQL_DAEMON_HPP_

#include "../cxx_ver.hpp"
#include <vector>
#include <deque>
#include <boost/shared_ptr.hpp>

namespace Poseidon {

namespace MySql {
	class ObjectBase;
	class Connection;
}

class JobPromise;

struct MySqlDaemon {
	struct SnapshotElement {
		unsigned thread;
		const char *table;
		unsigned long long ns_total;
	};

	static void start();
	static void stop();

	// 同步接口。
	static boost::shared_ptr<MySql::Connection> create_connection();

	// 异步接口。
	static std::vector<SnapshotElement> snapshot();

	static void wait_for_all_async_operations();

	// 以下第一个参数是出参。
	static boost::shared_ptr<const JobPromise> enqueue_for_saving(
		boost::shared_ptr<const MySql::ObjectBase> object, bool to_replace, bool urgent);
	static boost::shared_ptr<const JobPromise> enqueue_for_loading(
		boost::shared_ptr<MySql::ObjectBase> object, std::string query);
	static boost::shared_ptr<const JobPromise> enqueue_for_batch_loading(
		boost::shared_ptr<std::deque<boost::shared_ptr<MySql::ObjectBase> > > sink,
		boost::shared_ptr<MySql::ObjectBase> (*factory)(), const char *table_hint, std::string query);

private:
	MySqlDaemon();
};

}

#endif
