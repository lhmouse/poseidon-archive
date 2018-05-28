// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#ifndef POSEIDON_SINGLETONS_MYSQL_DAEMON_HPP_
#define POSEIDON_SINGLETONS_MYSQL_DAEMON_HPP_

#include "../cxx_ver.hpp"
#include "../mysql/fwd.hpp"
#include <boost/shared_ptr.hpp>
#include <boost/function.hpp>
#include <string>

namespace Poseidon {

class Promise;

class Mysql_daemon {
private:
	Mysql_daemon();

public:
	typedef std::function<void (const boost::shared_ptr<Mysql::Connection> &)> Query_callback;

	static void start();
	static void stop();

	// 同步接口。
	static boost::shared_ptr<Mysql::Connection> create_connection(bool from_slave = false);

	static void wait_for_all_async_operations();

	// 异步接口。
	static boost::shared_ptr<const Promise> enqueue_for_saving(boost::shared_ptr<const Mysql::Object_base> object, bool to_replace, bool urgent);
	static boost::shared_ptr<const Promise> enqueue_for_loading(boost::shared_ptr<Mysql::Object_base> object, std::string query);
	static boost::shared_ptr<const Promise> enqueue_for_deleting(const char *table_hint, std::string query);
	static boost::shared_ptr<const Promise> enqueue_for_batch_loading(Query_callback callback, const char *table_hint, std::string query);

	static void enqueue_for_low_level_access(const boost::shared_ptr<Promise> &promise, Query_callback callback, const char *table_hint, bool from_slave = false);

	static boost::shared_ptr<const Promise> enqueue_for_waiting_for_all_async_operations();
};

}

#endif
