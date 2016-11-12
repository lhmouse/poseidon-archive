// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2016, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_SINGLETONS_MYSQL_DAEMON_HPP_
#define POSEIDON_SINGLETONS_MYSQL_DAEMON_HPP_

#include "../cxx_ver.hpp"
#include <boost/shared_ptr.hpp>
#include <boost/function.hpp>
#include <string>

namespace Poseidon {

namespace MySql {
	class ObjectBase;
	class Connection;
}

class JobPromise;

class MySqlDaemon {
private:
	MySqlDaemon();

public:
	typedef boost::function<void (const boost::shared_ptr<MySql::Connection> &)> ObjectFactory;

	static void start();
	static void stop();

	// 同步接口。
	static boost::shared_ptr<MySql::Connection> create_connection(bool from_slave = false);

	static void wait_for_all_async_operations();

	// 异步接口。
	// 以下第一个参数是出参。
	static boost::shared_ptr<const JobPromise> enqueue_for_saving(
		boost::shared_ptr<const MySql::ObjectBase> object, bool to_replace, bool urgent);
	static boost::shared_ptr<const JobPromise> enqueue_for_loading(
		boost::shared_ptr<MySql::ObjectBase> object, std::string query);
	static boost::shared_ptr<const JobPromise> enqueue_for_deleting(
		const char *table_hint, std::string query);
	static boost::shared_ptr<const JobPromise> enqueue_for_batch_loading(
		ObjectFactory factory, const char *table_hint, std::string query);

	static boost::shared_ptr<const JobPromise> enqueue_for_waiting_for_all_async_operations();
};

}

#endif
