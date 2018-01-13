// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "mysql_daemon.hpp"
#include "main_config.hpp"
#include "../mysql/object_base.hpp"
#include "../mysql/exception.hpp"
#include "../mysql/connection.hpp"
#include "../thread.hpp"
#include "../mutex.hpp"
#include "../condition_variable.hpp"
#include "../atomic.hpp"
#include "../exception.hpp"
#include "../log.hpp"
#include "../raii.hpp"
#include "../promise.hpp"
#include "../profiler.hpp"
#include "../time.hpp"
#include "../errno.hpp"
#include "../buffer_streams.hpp"
#include "../checked_arithmetic.hpp"
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <mysql/mysqld_error.h>
#include <mysql/errmsg.h>

namespace Poseidon {

typedef MySqlDaemon::QueryCallback QueryCallback;

namespace {
	boost::shared_ptr<MySql::Connection> real_create_connection(bool from_slave, const boost::shared_ptr<MySql::Connection> &master_conn){
		std::string server_addr;
		boost::uint16_t server_port = 0;
		if(from_slave){
			server_addr = MainConfig::get<std::string>("mysql_slave_addr");
			server_port = MainConfig::get<boost::uint16_t>("mysql_slave_port");
		}
		if(server_addr.empty()){
			if(master_conn){
				LOG_POSEIDON_DEBUG("MySQL slave is not configured. Reuse the master connection as a slave.");
				return master_conn;
			}
			server_addr = MainConfig::get<std::string>("mysql_server_addr", "localhost");
			server_port = MainConfig::get<boost::uint16_t>("mysql_server_port", 3306);
		}
		std::string username = MainConfig::get<std::string>("mysql_username", "root");
		std::string password = MainConfig::get<std::string>("mysql_password");
		std::string schema = MainConfig::get<std::string>("mysql_schema", "poseidon");
		bool use_ssl = MainConfig::get<bool>("mysql_use_ssl", false);
		std::string charset = MainConfig::get<std::string>("mysql_charset", "utf8");
		return MySql::Connection::create(server_addr.c_str(), server_port, username.c_str(), password.c_str(), schema.c_str(), use_ssl, charset.c_str());
	}

	// 对于日志文件的写操作应当互斥。
	Mutex g_dump_mutex;

	void dump_sql_to_file(const std::string &query, unsigned long err_code, const char *err_msg) NOEXCEPT
	try {
		PROFILE_ME;

		const AUTO(dump_dir, MainConfig::get<std::string>("mysql_dump_dir"));
		if(dump_dir.empty()){
			LOG_POSEIDON_WARNING("MySQL dump is disabled.");
			return;
		}

		const AUTO(local_now, get_local_time());
		const AUTO(dt, break_down_time(local_now));
		char temp[256];
		std::size_t len = (unsigned)std::sprintf(temp, "%04u-%02u-%02u_%05u.log", dt.yr, dt.mon, dt.day, (unsigned)::getpid());
		std::string dump_path;
		dump_path.reserve(1023);
		dump_path.assign(dump_dir);
		dump_path.push_back('/');
		dump_path.append(temp, len);

		LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "Creating SQL dump file: ", dump_path);
		UniqueFile dump_file;
		if(!dump_file.reset(::open(dump_path.c_str(), O_WRONLY | O_APPEND | O_CREAT, 0644))){
			const int saved_errno = errno;
			LOG_POSEIDON_FATAL("Error creating SQL dump file: dump_path = ", dump_path, ", errno = ", saved_errno, ", desc = ", get_error_desc(saved_errno));
			std::abort();
		}

		LOG_POSEIDON_INFO("Writing MySQL dump...");
		Buffer_ostream os;
		len = format_time(temp, sizeof(temp), local_now, false);
		os <<"-- " <<temp <<": err_code = " <<err_code <<", err_msg = " <<err_msg <<std::endl;
		if(query.empty()){
			os <<"-- <low level access>";
		} else {
			os <<query <<";";
		}
		os <<std::endl <<std::endl;
		const AUTO(str, os.get_buffer().dump_string());

		const Mutex::UniqueLock lock(g_dump_mutex);
		std::size_t total = 0;
		do {
			::ssize_t written = ::write(dump_file.get(), str.data() + total, str.size() - total);
			if(written <= 0){
				break;
			}
			total += static_cast<std::size_t>(written);
		} while(total < str.size());
	} catch(std::exception &e){
		LOG_POSEIDON_ERROR("Error writing SQL dump: what = ", e.what());
	}

	// 数据库线程操作。
	class OperationBase : NONCOPYABLE {
	private:
		const boost::weak_ptr<Promise> m_weak_promise;

		boost::shared_ptr<const void> m_probe;

	public:
		explicit OperationBase(const boost::shared_ptr<Promise> &promise)
			: m_weak_promise(promise)
		{ }
		virtual ~OperationBase(){ }

	public:
		void set_probe(boost::shared_ptr<const void> probe){
			m_probe = STD_MOVE(probe);
		}

		virtual boost::shared_ptr<Promise> get_promise() const {
			return m_weak_promise.lock();
		}
		virtual bool should_use_slave() const = 0;
		virtual boost::shared_ptr<const MySql::ObjectBase> get_combinable_object() const = 0;
		virtual const char *get_table() const = 0;
		virtual void generate_sql(std::string &query) const = 0;
		virtual void execute(const boost::shared_ptr<MySql::Connection> &conn, const std::string &query) = 0;
	};

	class SaveOperation : public OperationBase {
	private:
		boost::shared_ptr<const MySql::ObjectBase> m_object;
		bool m_to_replace;

	public:
		SaveOperation(const boost::shared_ptr<Promise> &promise, boost::shared_ptr<const MySql::ObjectBase> object, bool to_replace)
			: OperationBase(promise)
			, m_object(STD_MOVE(object)), m_to_replace(to_replace)
		{ }

	protected:
		bool should_use_slave() const OVERRIDE {
			return false;
		}
		boost::shared_ptr<const MySql::ObjectBase> get_combinable_object() const OVERRIDE {
			return m_object;
		}
		const char *get_table() const OVERRIDE {
			return m_object->get_table();
		}
		void generate_sql(std::string &query) const OVERRIDE {
			Buffer_ostream os;
			if(m_to_replace){
				os <<"REPLACE";
			} else {
				os <<"INSERT";
			}
			os <<" INTO `" <<get_table() <<"` SET ";
			m_object->generate_sql(os);
			query = os.get_buffer().dump_string();
		}
		void execute(const boost::shared_ptr<MySql::Connection> &conn, const std::string &query) OVERRIDE {
			PROFILE_ME;

			conn->execute_sql(query);
		}
	};

	class LoadOperation : public OperationBase {
	private:
		boost::shared_ptr<MySql::ObjectBase> m_object;
		std::string m_query;

	public:
		LoadOperation(const boost::shared_ptr<Promise> &promise, boost::shared_ptr<MySql::ObjectBase> object, std::string query)
			: OperationBase(promise)
			, m_object(STD_MOVE(object)), m_query(STD_MOVE(query))
		{ }

	protected:
		bool should_use_slave() const OVERRIDE {
			return true;
		}
		boost::shared_ptr<const MySql::ObjectBase> get_combinable_object() const OVERRIDE {
			return VAL_INIT; // 不能合并。
		}
		const char *get_table() const OVERRIDE {
			return m_object->get_table();
		}
		void generate_sql(std::string &query) const OVERRIDE {
			query = m_query;
		}
		void execute(const boost::shared_ptr<MySql::Connection> &conn, const std::string &query) OVERRIDE {
			PROFILE_ME;

			if(!get_promise()){
				LOG_POSEIDON_DEBUG("Discarding isolated MySQL query: table = ", get_table(), ", query = ", query);
				return;
			}
			conn->execute_sql(query);
			DEBUG_THROW_UNLESS(conn->fetch_row(), MySql::Exception, SharedNts::view(get_table()), ER_SP_FETCH_NO_DATA, sslit("No rows returned"));
			m_object->fetch(conn);
		}
	};

	class DeleteOperation : public OperationBase {
	private:
		const char *m_table_hint;
		std::string m_query;

	public:
		DeleteOperation(const boost::shared_ptr<Promise> &promise, const char *table_hint, std::string query)
			: OperationBase(promise)
			, m_table_hint(table_hint), m_query(STD_MOVE(query))
		{ }

	protected:
		bool should_use_slave() const OVERRIDE {
			return false;
		}
		boost::shared_ptr<const MySql::ObjectBase> get_combinable_object() const OVERRIDE {
			return VAL_INIT; // 不能合并。
		}
		const char *get_table() const OVERRIDE {
			return m_table_hint;
		}
		void generate_sql(std::string &query) const OVERRIDE {
			query = m_query;
		}
		void execute(const boost::shared_ptr<MySql::Connection> &conn, const std::string &query) OVERRIDE {
			PROFILE_ME;

			conn->execute_sql(query);
		}
	};

	class BatchLoadOperation : public OperationBase {
	private:
		QueryCallback m_callback;
		const char *m_table_hint;
		std::string m_query;

	public:
		BatchLoadOperation(const boost::shared_ptr<Promise> &promise, QueryCallback callback, const char *table_hint, std::string query)
			: OperationBase(promise)
			, m_callback(STD_MOVE_IDN(callback)), m_table_hint(table_hint), m_query(STD_MOVE(query))
		{ }

	protected:
		bool should_use_slave() const OVERRIDE {
			return true;
		}
		boost::shared_ptr<const MySql::ObjectBase> get_combinable_object() const OVERRIDE {
			return VAL_INIT; // 不能合并。
		}
		const char *get_table() const OVERRIDE {
			return m_table_hint;
		}
		void generate_sql(std::string &query) const OVERRIDE {
			query = m_query;
		}
		void execute(const boost::shared_ptr<MySql::Connection> &conn, const std::string &query) OVERRIDE {
			PROFILE_ME;

			if(!get_promise()){
				LOG_POSEIDON_DEBUG("Discarding isolated MySQL query: table = ", get_table(), ", query = ", query);
				return;
			}
			conn->execute_sql(query);
			if(m_callback){
				while(conn->fetch_row()){
					m_callback(conn);
				}
			} else {
				LOG_POSEIDON_DEBUG("Result discarded.");
			}
		}
	};

	class LowLevelAccessOperation : public OperationBase {
	private:
		QueryCallback m_callback;
		const char *m_table_hint;
		bool m_from_slave;

	public:
		LowLevelAccessOperation(const boost::shared_ptr<Promise> &promise, QueryCallback callback, const char *table_hint, bool from_slave)
			: OperationBase(promise)
			, m_callback(STD_MOVE_IDN(callback)), m_table_hint(table_hint), m_from_slave(from_slave)
		{ }

	protected:
		bool should_use_slave() const OVERRIDE {
			return m_from_slave;
		}
		boost::shared_ptr<const MySql::ObjectBase> get_combinable_object() const OVERRIDE {
			return VAL_INIT; // 不能合并。
		}
		const char *get_table() const OVERRIDE {
			return m_table_hint;
		}
		void generate_sql(std::string & /* query */) const OVERRIDE {
			// no query
		}
		void execute(const boost::shared_ptr<MySql::Connection> &conn, const std::string & /* query */) OVERRIDE {
			PROFILE_ME;

			m_callback(conn);
		}
	};

	class WaitOperation : public OperationBase {
	public:
		explicit WaitOperation(const boost::shared_ptr<Promise> &promise)
			: OperationBase(promise)
		{ }
		~WaitOperation() OVERRIDE {
			const AUTO(promise, OperationBase::get_promise());
			if(promise){
				promise->set_success(false);
			}
		}

	protected:
		boost::shared_ptr<Promise> get_promise() const OVERRIDE {
			return VAL_INIT;
		}
		bool should_use_slave() const OVERRIDE {
			return false;
		}
		boost::shared_ptr<const MySql::ObjectBase> get_combinable_object() const OVERRIDE {
			return VAL_INIT; // 不能合并。
		}
		const char *get_table() const OVERRIDE {
			return "";
		}
		void generate_sql(std::string &query) const OVERRIDE {
			query = "DO 0";
		}
		void execute(const boost::shared_ptr<MySql::Connection> &conn, const std::string &query) OVERRIDE {
			PROFILE_ME;

			conn->execute_sql(query);
		}
	};

	class MySqlThread : NONCOPYABLE {
	private:
		struct OperationQueueElement {
			boost::shared_ptr<OperationBase> operation;
			boost::uint64_t due_time;
			std::size_t retry_count;
		};

	private:
		Thread m_thread;
		volatile bool m_running;

		mutable Mutex m_mutex;
		mutable ConditionVariable m_new_operation;
		volatile bool m_urgent; // 无视延迟写入，一次性处理队列中所有操作。
		boost::container::deque<OperationQueueElement> m_queue;

	public:
		MySqlThread()
			: m_running(false)
			, m_urgent(false)
		{ }

	private:
		bool pump_one_operation(boost::shared_ptr<MySql::Connection> &master_conn, boost::shared_ptr<MySql::Connection> &slave_conn) NOEXCEPT {
			PROFILE_ME;

			const AUTO(now, get_fast_mono_clock());
			OperationQueueElement *elem;
			{
				const Mutex::UniqueLock lock(m_mutex);
				if(m_queue.empty()){
					atomic_store(m_urgent, false, ATOMIC_RELAXED);
					return false;
				}
				if(!atomic_load(m_urgent, ATOMIC_CONSUME) && (now < m_queue.front().due_time)){
					return false;
				}
				elem = &m_queue.front();
			}
			const AUTO_REF(operation, elem->operation);
			AUTO_REF(conn, elem->operation->should_use_slave() ? slave_conn : master_conn);

			std::string query;
			STD_EXCEPTION_PTR except;
			unsigned long err_code = 0;
			char err_msg[4096];
			err_msg[0] = 0;

			bool execute_it = false;
			const AUTO(combinable_object, elem->operation->get_combinable_object());
			if(!combinable_object){
				execute_it = true;
			} else {
				const AUTO(old_write_stamp, combinable_object->get_combined_write_stamp());
				if(!old_write_stamp){
					execute_it = true;
				} else if(old_write_stamp == elem){
					combinable_object->set_combined_write_stamp(NULLPTR);
					execute_it = true;
				}
			}
			if(execute_it){
				try {
					operation->generate_sql(query);
					LOG_POSEIDON_DEBUG("Executing SQL: table = ", operation->get_table(), ", query = ", query);
					operation->execute(conn, query);
				} catch(MySql::Exception &e){
					LOG_POSEIDON_WARNING("MySql::Exception thrown: code = ", e.get_code(), ", what = ", e.what());
					except = STD_CURRENT_EXCEPTION();
					err_code = e.get_code();
					::snprintf(err_msg, sizeof(err_msg), "MySql::Exception: %s", e.what());
				} catch(std::exception &e){
					LOG_POSEIDON_WARNING("std::exception thrown: what = ", e.what());
					except = STD_CURRENT_EXCEPTION();
					err_code = ER_UNKNOWN_ERROR;
					::snprintf(err_msg, sizeof(err_msg), "std::exception: %s", e.what());
				} catch(...){
					LOG_POSEIDON_WARNING("Unknown exception thrown");
					except = STD_CURRENT_EXCEPTION();
					err_code = ER_UNKNOWN_ERROR;
					::strcpy(err_msg, "Unknown exception");
				}
				conn->discard_result();
			}
			if(except){
				const AUTO(max_retry_count, MainConfig::get<std::size_t>("mysql_max_retry_count", 3));
				const AUTO(retry_count, ++(elem->retry_count));
				if(retry_count < max_retry_count){
					LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "Going to retry MySQL operation: retry_count = ", retry_count);
					const AUTO(retry_init_delay, MainConfig::get<boost::uint64_t>("mysql_retry_init_delay", 1000));
					elem->due_time = now + (retry_init_delay << retry_count);
					conn.reset();
					return true;
				}
				LOG_POSEIDON_ERROR("Max retry count exceeded.");
				dump_sql_to_file(query, err_code, err_msg);
			}
			const AUTO(promise, elem->operation->get_promise());
			if(promise){
				if(except){
					promise->set_success(false);
				} else {
					promise->set_exception(STD_MOVE(except), false);
				}
			}
			const Mutex::UniqueLock lock(m_mutex);
			m_queue.pop_front();
			return true;
		}

		void thread_proc(){
			PROFILE_ME;
			LOG_POSEIDON_INFO("MySQL thread started.");

			boost::shared_ptr<MySql::Connection> master_conn, slave_conn;
			unsigned timeout = 0;
			for(;;){
				const AUTO(reconnect_delay, MainConfig::get<boost::uint64_t>("mysql_reconn_delay", 5000));
				bool busy;
				do {
					while(!master_conn){
						LOG_POSEIDON_INFO("Connecting to MySQL master server...");
						try {
							master_conn = real_create_connection(false, VAL_INIT);
							LOG_POSEIDON_INFO("Successfully connected to MySQL master server.");
						} catch(std::exception &e){
							LOG_POSEIDON_ERROR("std::exception thrown: what = ", e.what());
							::timespec req;
							req.tv_sec = (::time_t)(reconnect_delay / 1000);
							req.tv_nsec = (long)(reconnect_delay % 1000) * 1000 * 1000;
							::nanosleep(&req, NULLPTR);
						}
					}
					while(!slave_conn){
						LOG_POSEIDON_INFO("Connecting to MySQL slave server...");
						try {
							slave_conn = real_create_connection(true, master_conn);
							LOG_POSEIDON_INFO("Successfully connected to MySQL slave server.");
						} catch(std::exception &e){
							LOG_POSEIDON_ERROR("std::exception thrown: what = ", e.what());
							::timespec req;
							req.tv_sec = (::time_t)(reconnect_delay / 1000);
							req.tv_nsec = (long)(reconnect_delay % 1000) * 1000 * 1000;
							::nanosleep(&req, NULLPTR);
						}
					}
					busy = pump_one_operation(master_conn, slave_conn);
					timeout = std::min<unsigned>(timeout * 2u + 1u, !busy * 100u);
				} while(busy);

				Mutex::UniqueLock lock(m_mutex);
				if(m_queue.empty() && !atomic_load(m_running, ATOMIC_CONSUME)){
					break;
				}
				m_new_operation.timed_wait(lock, timeout);
			}

			LOG_POSEIDON_INFO("MySQL thread stopped.");
		}

	public:
		void start(){
			const Mutex::UniqueLock lock(m_mutex);
			Thread(boost::bind(&MySqlThread::thread_proc, this), sslit(" M  "), sslit("MySQL")).swap(m_thread);
			atomic_store(m_running, true, ATOMIC_RELEASE);
		}
		void stop(){
			atomic_store(m_running, false, ATOMIC_RELEASE);
		}
		void safe_join(){
			wait_till_idle();

			if(m_thread.joinable()){
				m_thread.join();
			}
		}

		void wait_till_idle(){
			for(;;){
				std::size_t pending_objects;
				std::string current_sql;
				{
					const Mutex::UniqueLock lock(m_mutex);
					pending_objects = m_queue.size();
					if(pending_objects == 0){
						break;
					}
					m_queue.front().operation->generate_sql(current_sql);
					atomic_store(m_urgent, true, ATOMIC_RELEASE);
					m_new_operation.signal();
				}
				LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "Waiting for SQL queries to complete: pending_objects = ", pending_objects, ", current_sql = ", current_sql);

				::timespec req;
				req.tv_sec = 0;
				req.tv_nsec = 500 * 1000 * 1000;
				::nanosleep(&req, NULLPTR);
			}
		}

		std::size_t get_queue_size() const {
			const Mutex::UniqueLock lock(m_mutex);
			return m_queue.size();
		}
		void add_operation(boost::shared_ptr<OperationBase> operation, bool urgent){
			PROFILE_ME;

			const AUTO(combinable_object, operation->get_combinable_object());

			const AUTO(now, get_fast_mono_clock());
			const AUTO(save_delay, MainConfig::get<boost::uint64_t>("mysql_save_delay", 5000));
			// 有紧急操作时无视写入延迟，这个逻辑不在这里处理。
			const AUTO(due_time, saturated_add(now, save_delay));

			const Mutex::UniqueLock lock(m_mutex);
			DEBUG_THROW_UNLESS(atomic_load(m_running, ATOMIC_CONSUME), Exception, sslit("MySQL thread is being shut down"));
			OperationQueueElement elem = { STD_MOVE(operation), due_time };
			m_queue.push_back(STD_MOVE(elem));
			if(combinable_object){
				const AUTO(old_write_stamp, combinable_object->get_combined_write_stamp());
				if(!old_write_stamp){
					combinable_object->set_combined_write_stamp(&m_queue.back());
				}
			}
			if(urgent){
				atomic_store(m_urgent, true, ATOMIC_RELEASE);
			}
			m_new_operation.signal();
		}
	};

	volatile bool g_running = false;

	Mutex g_router_mutex;
	struct Route {
		boost::shared_ptr<const void> probe;
		boost::shared_ptr<MySqlThread> thread;
	};
	boost::container::flat_map<SharedNts, Route> g_router;
	boost::container::flat_multimap<std::size_t, std::size_t> g_routing_map;
	boost::container::vector<boost::shared_ptr<MySqlThread> > g_threads;

	void add_operation_by_table(const char *table, boost::shared_ptr<OperationBase> operation, bool urgent){
		PROFILE_ME;
		DEBUG_THROW_UNLESS(!g_threads.empty(), BasicException, sslit("MySQL support is not enabled"));

		boost::shared_ptr<const void> probe;
		boost::shared_ptr<MySqlThread> thread;
		{
			const Mutex::UniqueLock lock(g_router_mutex);

			AUTO_REF(route, g_router[SharedNts::view(table)]);
			if(route.probe.use_count() > 1){
				probe = route.probe;
				thread = route.thread;
				goto _use_thread;
			}
			if(!route.probe){
				route.probe = boost::make_shared<int>();
			}
			probe = route.probe;

			g_routing_map.clear();
			g_routing_map.reserve(g_threads.size());
			for(std::size_t i = 0; i < g_threads.size(); ++i){
				AUTO_REF(test_thread, g_threads.at(i));
				if(!test_thread){
					LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_DEBUG, "Creating new MySQL thread ", i, " for table ", table);
					thread = boost::make_shared<MySqlThread>();
					thread->start();
					test_thread = thread;
					route.thread = thread;
					goto _use_thread;
				}
				const AUTO(queue_size, test_thread->get_queue_size());
				LOG_POSEIDON_DEBUG("> MySQL thread ", i, "'s queue size: ", queue_size);
				g_routing_map.emplace(queue_size, i);
			}
			if(g_routing_map.empty()){
				LOG_POSEIDON_FATAL("No available MySQL thread?!");
				std::abort();
			}
			const AUTO(index, g_routing_map.begin()->second);
			LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_DEBUG, "Picking thread ", index, " for table ", table);
			thread = g_threads.at(index);
			route.thread = thread;
		}
	_use_thread:
		assert(probe);
		assert(thread);
		operation->set_probe(STD_MOVE(probe));
		thread->add_operation(STD_MOVE(operation), urgent);
	}
	void add_operation_all(boost::shared_ptr<OperationBase> operation, bool urgent){
		PROFILE_ME;
		DEBUG_THROW_UNLESS(!g_threads.empty(), BasicException, sslit("MySQL support is not enabled"));

		const Mutex::UniqueLock lock(g_router_mutex);
		for(AUTO(it, g_threads.begin()); it != g_threads.end(); ++it){
			const AUTO_REF(thread, *it);
			if(!thread){
				continue;
			}
			thread->add_operation(operation, urgent);
		}
	}
}

void MySqlDaemon::start(){
	if(atomic_exchange(g_running, true, ATOMIC_ACQ_REL) != false){
		LOG_POSEIDON_FATAL("Only one daemon is allowed at the same time.");
		std::abort();
	}
	LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "Starting MySQL daemon...");

	const AUTO(max_thread_count, MainConfig::get<std::size_t>("mysql_max_thread_count"));
	if(max_thread_count == 0){
		LOG_POSEIDON_WARNING("MySQL support has been disabled. To enable MySQL support, set `mysql_max_thread_count` in `main.conf` to a value greater than zero.");
	} else {
		boost::shared_ptr<MySql::Connection> master_conn, slave_conn;
		LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "Checking whether MySQL master server is up...");
		try {
			master_conn = real_create_connection(false, VAL_INIT);
			master_conn->execute_sql("DO 0");
		} catch(std::exception &e){
			LOG_POSEIDON_FATAL("Could not connect to MySQL master server: ", e.what());
			LOG_POSEIDON_WARNING("To disable MySQL support, set `mysql_max_thread_count` in `main.conf` to zero.");
			std::abort();
		}

		LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "Checking whether MySQL slave server is up...");
		try {
			slave_conn = real_create_connection(true, master_conn);
			if(slave_conn != master_conn){
				slave_conn->execute_sql("DO 0");
			}
		} catch(std::exception &e){
			LOG_POSEIDON_FATAL("Could not connect to MySQL slave server: ", e.what());
			LOG_POSEIDON_WARNING("To disable MySQL support, set `mysql_max_thread_count` in `main.conf` to zero.");
			std::abort();
		}

		const AUTO(dump_dir, MainConfig::get<std::string>("mysql_dump_dir"));
		if(dump_dir.empty()){
			LOG_POSEIDON_WARNING("MySQL error dump has been disabled. To enable MySQL error dump, set `mysql_dump_dir` in `main.conf` to the path to the dump directory.");
		} else {
			LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "Checking whether MySQL dump directory is writeable...");
			try {
				const AUTO(placeholder_path, dump_dir + "/placeholder");
				DEBUG_THROW_ASSERT(UniqueFile(::open(placeholder_path.c_str(), O_WRONLY | O_TRUNC | O_CREAT, 0644)));
			} catch(std::exception &e){
				LOG_POSEIDON_FATAL("Could not write MySQL dump: ", e.what());
				LOG_POSEIDON_WARNING("To disable MySQL error dump, set `mysql_dump_dir` in `main.conf` to an empty string.");
				std::abort();
			}
		}
	}
	g_threads.resize(max_thread_count);

	LOG_POSEIDON_INFO("MySQL daemon started.");
}
void MySqlDaemon::stop(){
	if(atomic_exchange(g_running, false, ATOMIC_ACQ_REL) == false){
		return;
	}
	LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "Stopping MySQL daemon...");

	for(std::size_t i = 0; i < g_threads.size(); ++i){
		const AUTO_REF(thread, g_threads.at(i));
		if(!thread){
			continue;
		}
		LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "Stopping MySQL thread ", i);
		thread->stop();
	}
	for(std::size_t i = 0; i < g_threads.size(); ++i){
		const AUTO_REF(thread, g_threads.at(i));
		if(!thread){
			continue;
		}
		LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "Waiting for MySQL thread ", i, " to terminate...");
		thread->safe_join();
	}
	g_threads.clear();

	LOG_POSEIDON_INFO("MySQL daemon stopped.");
}

boost::shared_ptr<MySql::Connection> MySqlDaemon::create_connection(bool from_slave){
	return real_create_connection(from_slave, VAL_INIT);
}

void MySqlDaemon::wait_for_all_async_operations(){
	for(std::size_t i = 0; i < g_threads.size(); ++i){
		const AUTO_REF(thread, g_threads.at(i));
		if(!thread){
			continue;
		}
		thread->wait_till_idle();
	}
}

boost::shared_ptr<const Promise> MySqlDaemon::enqueue_for_saving(boost::shared_ptr<const MySql::ObjectBase> object, bool to_replace, bool urgent){
	AUTO(promise, boost::make_shared<Promise>());
	const char *const table = object->get_table();
	AUTO(operation, boost::make_shared<SaveOperation>(promise, STD_MOVE(object), to_replace));
	add_operation_by_table(table, STD_MOVE_IDN(operation), urgent);
	return STD_MOVE_IDN(promise);
}
boost::shared_ptr<const Promise> MySqlDaemon::enqueue_for_loading(boost::shared_ptr<MySql::ObjectBase> object, std::string query){
	DEBUG_THROW_ASSERT(!query.empty());

	AUTO(promise, boost::make_shared<Promise>());
	const char *const table = object->get_table();
	AUTO(operation, boost::make_shared<LoadOperation>(promise, STD_MOVE(object), STD_MOVE(query)));
	add_operation_by_table(table, STD_MOVE_IDN(operation), true);
	return STD_MOVE_IDN(promise);
}
boost::shared_ptr<const Promise> MySqlDaemon::enqueue_for_deleting(const char *table_hint, std::string query){
	DEBUG_THROW_ASSERT(!query.empty());

	AUTO(promise, boost::make_shared<Promise>());
	const char *const table = table_hint;
	AUTO(operation, boost::make_shared<DeleteOperation>(promise, table_hint, STD_MOVE(query)));
	add_operation_by_table(table, STD_MOVE_IDN(operation), true);
	return STD_MOVE_IDN(promise);
}
boost::shared_ptr<const Promise> MySqlDaemon::enqueue_for_batch_loading(QueryCallback callback, const char *table_hint, std::string query){
	DEBUG_THROW_ASSERT(!query.empty());

	AUTO(promise, boost::make_shared<Promise>());
	const char *const table = table_hint;
	AUTO(operation, boost::make_shared<BatchLoadOperation>(promise, STD_MOVE(callback), table_hint, STD_MOVE(query)));
	add_operation_by_table(table, STD_MOVE_IDN(operation), true);
	return STD_MOVE_IDN(promise);
}

void MySqlDaemon::enqueue_for_low_level_access(const boost::shared_ptr<Promise> &promise, QueryCallback callback, const char *table_hint, bool from_slave){
	const char *const table = table_hint;
	AUTO(operation, boost::make_shared<LowLevelAccessOperation>(promise, STD_MOVE(callback), table_hint, from_slave));
	add_operation_by_table(table, STD_MOVE_IDN(operation), true);
}

boost::shared_ptr<const Promise> MySqlDaemon::enqueue_for_waiting_for_all_async_operations(){
	AUTO(promise, boost::make_shared<Promise>());
	AUTO(operation, boost::make_shared<WaitOperation>(promise));
	add_operation_all(STD_MOVE_IDN(operation), true);
	return STD_MOVE_IDN(promise);
}

}
