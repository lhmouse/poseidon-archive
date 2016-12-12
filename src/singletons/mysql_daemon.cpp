// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2016, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "mysql_daemon.hpp"
#include "main_config.hpp"
#include <boost/container/flat_map.hpp>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <mysqld_error.h>
#include <errmsg.h>
#include "../mysql/object_base.hpp"
#include "../mysql/exception.hpp"
#include "../mysql/connection.hpp"
#include "../mysql/thread_context.hpp"
#include "../thread.hpp"
#include "../mutex.hpp"
#include "../condition_variable.hpp"
#include "../atomic.hpp"
#include "../exception.hpp"
#include "../log.hpp"
#include "../raii.hpp"
#include "../job_promise.hpp"
#include "../profiler.hpp"
#include "../time.hpp"
#include "../errno.hpp"
#include "../buffer_streams.hpp"

namespace Poseidon {

namespace {
	std::string     g_master_addr       = "localhost";
	unsigned        g_master_port       = 3306;
	std::string     g_slave_addr        = VAL_INIT;
	unsigned        g_slave_port        = 0;
	std::string     g_username          = "root";
	std::string     g_password          = "root";
	std::string     g_schema            = "poseidon";
	bool            g_use_ssl           = false;
	std::string     g_charset           = "utf8";

	std::string     g_dump_dir          = VAL_INIT;
	boost::uint64_t g_save_delay        = 5000;
	boost::uint64_t g_reconn_delay      = 10000;
	std::size_t     g_max_retry_count   = 3;
	boost::uint64_t g_retry_init_delay  = 1000;
	std::size_t     g_max_thread_count  = 8;

	// 对于日志文件的写操作应当互斥。
	Mutex g_dump_mutex;

	// 数据库线程操作。
	class OperationBase : NONCOPYABLE {
	private:
		boost::shared_ptr<const void> m_probe;

	public:
		virtual ~OperationBase(){
		}

	public:
		void set_probe(boost::shared_ptr<const void> probe){
			m_probe = STD_MOVE(probe);
		}

		virtual bool should_use_slave() const = 0;
		virtual boost::shared_ptr<const MySql::ObjectBase> get_combinable_object() const = 0;
		virtual const char *get_table_name() const = 0;
		virtual void generate_sql(std::string &query) const = 0;
		virtual void execute(const boost::shared_ptr<MySql::Connection> &conn, const std::string &query) const = 0;
		virtual void set_success() = 0;
#ifdef POSEIDON_CXX11
		virtual void set_exception(std::exception_ptr ep) = 0;
#else
		virtual void set_exception(boost::exception_ptr ep) = 0;
#endif
	};

	class SaveOperation : public OperationBase {
	private:
		const boost::shared_ptr<JobPromise> m_promise;
		const boost::shared_ptr<const MySql::ObjectBase> m_object;
		const bool m_to_replace;

	public:
		SaveOperation(boost::shared_ptr<JobPromise> promise,
			boost::shared_ptr<const MySql::ObjectBase> object, bool to_replace)
			: m_promise(STD_MOVE(promise)), m_object(STD_MOVE(object)), m_to_replace(to_replace)
		{
		}

	protected:
		bool should_use_slave() const {
			return false;
		}
		boost::shared_ptr<const MySql::ObjectBase> get_combinable_object() const OVERRIDE {
			return m_object;
		}
		const char *get_table_name() const OVERRIDE {
			return m_object->get_table_name();
		}
		void generate_sql(std::string &query) const OVERRIDE {
			Buffer_ostream os;
			if(m_to_replace){
				os <<"REPLACE";
			} else {
				os <<"INSERT";
			}
			os <<" INTO `" <<get_table_name() <<"` SET ";
			m_object->generate_sql(os);
			query = os.get_buffer().dump_string();
		}
		void execute(const boost::shared_ptr<MySql::Connection> &conn, const std::string &query) const OVERRIDE {
			PROFILE_ME;

			conn->execute_sql(query);
		}
		void set_success() OVERRIDE {
			m_promise->set_success();
		}
#ifdef POSEIDON_CXX11
		void set_exception(std::exception_ptr ep) override
#else
		void set_exception(boost::exception_ptr ep)
#endif
		{
			m_promise->set_exception(STD_MOVE(ep));
		}
	};

	class LoadOperation : public OperationBase {
	private:
		const boost::shared_ptr<JobPromise> m_promise;
		const boost::shared_ptr<MySql::ObjectBase> m_object;
		const std::string m_query;

	public:
		LoadOperation(boost::shared_ptr<JobPromise> promise,
			boost::shared_ptr<MySql::ObjectBase> object, std::string query)
			: m_promise(STD_MOVE(promise)), m_object(STD_MOVE(object)), m_query(STD_MOVE(query))
		{
		}

	protected:
		bool should_use_slave() const {
			return true;
		}
		boost::shared_ptr<const MySql::ObjectBase> get_combinable_object() const OVERRIDE {
			return VAL_INIT; // 不能合并。
		}
		const char *get_table_name() const OVERRIDE {
			return m_object->get_table_name();
		}
		void generate_sql(std::string &query) const OVERRIDE {
			query = m_query;
		}
		void execute(const boost::shared_ptr<MySql::Connection> &conn, const std::string &query) const OVERRIDE {
			PROFILE_ME;

			if(m_promise.unique()){
				LOG_POSEIDON_DEBUG("Discarding isolated MySQL query: table_name = ", get_table_name(), ", query = ", query);
				return;
			}

			conn->execute_sql(query);
			if(!conn->fetch_row()){
				DEBUG_THROW(MySql::Exception, SharedNts::view(get_table_name()), ER_SP_FETCH_NO_DATA, sslit("No rows returned"));
			}
			m_object->fetch(conn);
		}
		void set_success() OVERRIDE {
			m_promise->set_success();
		}
#ifdef POSEIDON_CXX11
		void set_exception(std::exception_ptr ep) override
#else
		void set_exception(boost::exception_ptr ep)
#endif
		{
			m_promise->set_exception(STD_MOVE(ep));
		}
	};

	class DeleteOperation : public OperationBase {
	private:
		const boost::shared_ptr<JobPromise> m_promise;
		const char *const m_table_hint;
		const std::string m_query;

	public:
		DeleteOperation(boost::shared_ptr<JobPromise> promise,
			const char *table_hint, std::string query)
			: m_promise(STD_MOVE(promise)), m_table_hint(table_hint), m_query(STD_MOVE(query))
		{
		}

	protected:
		bool should_use_slave() const {
			return false;
		}
		boost::shared_ptr<const MySql::ObjectBase> get_combinable_object() const OVERRIDE {
			return VAL_INIT; // 不能合并。
		}
		const char *get_table_name() const OVERRIDE {
			return m_table_hint;
		}
		void generate_sql(std::string &query) const OVERRIDE {
			query = m_query;
		}
		void execute(const boost::shared_ptr<MySql::Connection> &conn, const std::string &query) const OVERRIDE {
			PROFILE_ME;

			conn->execute_sql(query);
		}
		void set_success() OVERRIDE {
			m_promise->set_success();
		}
#ifdef POSEIDON_CXX11
		void set_exception(std::exception_ptr ep) override
#else
		void set_exception(boost::exception_ptr ep)
#endif
		{
			m_promise->set_exception(STD_MOVE(ep));
		}
	};

	class BatchLoadOperation : public OperationBase {
	private:
		const boost::shared_ptr<JobPromise> m_promise;
		const MySqlDaemon::ObjectFactory m_factory;
		const char *const m_table_hint;
		const std::string m_query;

	public:
		BatchLoadOperation(boost::shared_ptr<JobPromise> promise,
			MySqlDaemon::ObjectFactory factory, const char *table_hint, std::string query)
			: m_promise(STD_MOVE(promise)), m_factory(STD_MOVE_IDN(factory)), m_table_hint(table_hint), m_query(STD_MOVE(query))
		{
		}

	protected:
		bool should_use_slave() const {
			return true;
		}
		boost::shared_ptr<const MySql::ObjectBase> get_combinable_object() const OVERRIDE {
			return VAL_INIT; // 不能合并。
		}
		const char *get_table_name() const OVERRIDE {
			return m_table_hint;
		}
		void generate_sql(std::string &query) const OVERRIDE {
			query = m_query;
		}
		void execute(const boost::shared_ptr<MySql::Connection> &conn, const std::string &query) const OVERRIDE {
			PROFILE_ME;

			if(m_promise.unique()){
				LOG_POSEIDON_DEBUG("Discarding isolated MySQL query: table_name = ", get_table_name(), ", query = ", query);
				return;
			}

			conn->execute_sql(query);
			if(m_factory){
				while(conn->fetch_row()){
					m_factory(conn);
				}
			} else {
				LOG_POSEIDON_DEBUG("Result discarded.");
			}
		}
		void set_success() OVERRIDE {
			m_promise->set_success();
		}
#ifdef POSEIDON_CXX11
		void set_exception(std::exception_ptr ep) override
#else
		void set_exception(boost::exception_ptr ep)
#endif
		{
			m_promise->set_exception(STD_MOVE(ep));
		}
	};

	class WaitOperation : public OperationBase {
	private:
		const boost::shared_ptr<JobPromise> m_promise;

	public:
		explicit WaitOperation(boost::shared_ptr<JobPromise> promise)
			: m_promise(STD_MOVE(promise))
		{
		}
		~WaitOperation(){
			try {
				if(!m_promise->is_satisfied()){
					m_promise->set_success();
				}
			} catch(std::exception &e){
				LOG_POSEIDON_ERROR("std::exception thrown: what = ", e.what());
			} catch(...){
				LOG_POSEIDON_ERROR("Unknown exception thrown");
			}
		}

	protected:
		bool should_use_slave() const {
			return false;
		}
		boost::shared_ptr<const MySql::ObjectBase> get_combinable_object() const OVERRIDE {
			return VAL_INIT; // 不能合并。
		}
		const char *get_table_name() const OVERRIDE {
			return "";
		}
		void generate_sql(std::string &query) const OVERRIDE {
			query = "DO 0";
		}
		void execute(const boost::shared_ptr<MySql::Connection> &conn, const std::string &query) const OVERRIDE {
			PROFILE_ME;

			conn->execute_sql(query);
		}
		void set_success() OVERRIDE {
			// no-op
		}
#ifdef POSEIDON_CXX11
		void set_exception(std::exception_ptr ep) override
#else
		void set_exception(boost::exception_ptr ep)
#endif
		{
			m_promise->set_exception(STD_MOVE(ep));
		}
	};

	class MySqlThread : NONCOPYABLE {
	private:
		struct OperationQueueElement {
			boost::shared_ptr<OperationBase> operation;
			boost::uint64_t due_time;
			std::size_t retry_count;

			OperationQueueElement(boost::shared_ptr<OperationBase> operation_, boost::uint64_t due_time_)
				: operation(STD_MOVE(operation_)), due_time(due_time_), retry_count(0)
			{
			}
		};

	private:
		Thread m_thread;
		volatile bool m_running;
		volatile bool m_alive;

		mutable Mutex m_mutex;
		mutable ConditionVariable m_new_operation;
		volatile bool m_urgent; // 无视延迟写入，一次性处理队列中所有操作。
		boost::container::deque<OperationQueueElement> m_queue;

	public:
		MySqlThread()
			: m_running(false), m_alive(false)
			, m_urgent(false)
		{
		}

	private:
		void sleep_for_reconnection(){
			::timespec req;
			req.tv_sec = (::time_t)(g_reconn_delay / 1000);
			req.tv_nsec = (long)(g_reconn_delay % 1000) * 1000 * 1000;
			::nanosleep(&req, NULLPTR);
		}

		void thread_proc(){
			PROFILE_ME;
			LOG_POSEIDON_INFO("MySQL thread started.");

			const MySql::ThreadContext thread_context;
			boost::shared_ptr<MySql::Connection> master_conn, slave_conn;

			for(;;){
				while(!master_conn){
					LOG_POSEIDON_INFO("Connecting to MySQL master server...");
					try {
						master_conn = MySql::Connection::create(g_master_addr, g_master_port,
							g_username, g_password, g_schema, g_use_ssl, g_charset);
						LOG_POSEIDON_INFO("Successfully connected to MySQL master server.");
					} catch(std::exception &e){
						LOG_POSEIDON_ERROR("std::exception thrown: what = ", e.what());
						sleep_for_reconnection();
					}
				}
				if((g_master_addr == g_slave_addr) && (g_master_port == g_slave_port)){
					LOG_POSEIDON_TRACE("Reusing the master connection as the slave connection.");
					slave_conn = master_conn;
				} else {
					while(!slave_conn){
						LOG_POSEIDON_INFO("Connecting to MySQL slave server...");
						try {
							slave_conn = MySql::Connection::create(g_slave_addr, g_slave_port,
								g_username, g_password, g_schema, g_use_ssl, g_charset);
							LOG_POSEIDON_INFO("Successfully connected to MySQL slave server.");
						} catch(std::exception &e){
							LOG_POSEIDON_ERROR("std::exception thrown: what = ", e.what());
							sleep_for_reconnection();
						}
					}
				}

				bool should_retry = true;
				try {
					really_pump_operations(master_conn, slave_conn);
					should_retry = false;
				} catch(MySql::Exception &e){
					LOG_POSEIDON_WARNING("MySql::Exception thrown: schema = ", e.get_schema(),
						", code = ", e.get_code(), ", what = ", e.what());
				} catch(std::exception &e){
					LOG_POSEIDON_WARNING("std::exception thrown: what = ", e.what());
				} catch(...){
					LOG_POSEIDON_WARNING("Unknown exception thrown");
				}
				if(should_retry){
					master_conn.reset();
					slave_conn.reset();
					continue;
				}

				Mutex::UniqueLock lock(m_mutex);
				if(!atomic_load(m_running, ATOMIC_CONSUME) && m_queue.empty()){
					break;
				}
				m_new_operation.timed_wait(lock, 100);
			}

			atomic_store(m_alive, false, ATOMIC_RELEASE);
			LOG_POSEIDON_INFO("MySQL thread stopped.");
		}

		void really_pump_operations(const boost::shared_ptr<MySql::Connection> &master_conn,
			const boost::shared_ptr<MySql::Connection> &slave_conn)
		{
			PROFILE_ME;

			for(;;){
				const AUTO(now, get_fast_mono_clock());

				OperationQueueElement *elem;
				{
					const Mutex::UniqueLock lock(m_mutex);
					if(m_queue.empty()){
						atomic_store(m_urgent, false, ATOMIC_RELAXED);
						break;
					}
					if(!atomic_load(m_urgent, ATOMIC_CONSUME) && (now < m_queue.front().due_time)){
						break;
					}
					elem = &m_queue.front();
				}

				const bool uses_slave_conn = elem->operation->should_use_slave();
				const AUTO_REF(conn, uses_slave_conn ? slave_conn : master_conn);

				const AUTO_REF(operation, elem->operation);
#ifdef POSEIDON_CXX11
				std::exception_ptr except;
#else
				boost::exception_ptr except;
#endif

				std::string query;
				long err_code = 0;
				std::string err_msg;

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
					operation->generate_sql(query);
					try {
						LOG_POSEIDON_DEBUG("Executing SQL: table_name = ", operation->get_table_name(), ", query = ", query);
						operation->execute(conn, query);
					} catch(MySql::Exception &e){
						LOG_POSEIDON_WARNING("MySql::Exception thrown: code = ", e.get_code(), ", what = ", e.what());
#ifdef POSEIDON_CXX11
						except = std::current_exception();
#else
						except = boost::copy_exception(e);
#endif

						err_code = e.get_code();
						err_msg = e.what();
					} catch(std::exception &e){
						LOG_POSEIDON_WARNING("std::exception thrown: what = ", e.what());
#ifdef POSEIDON_CXX11
						except = std::current_exception();
#else
						except = boost::copy_exception(std::runtime_error(e.what()));
#endif

						err_code = ER_UNKNOWN_ERROR;
						err_msg = e.what();
					} catch(...){
						LOG_POSEIDON_WARNING("Unknown exception thrown");
#ifdef POSEIDON_CXX11
						except = std::current_exception();
#else
						except = boost::copy_exception(std::bad_exception());
#endif

						err_code = ER_UNKNOWN_ERROR;
						err_msg = "Unknown exception";
					}
					conn->discard_result();
				}

				if(except){
					const AUTO(retry_count, ++elem->retry_count);
					if(retry_count < g_max_retry_count){
						LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO,
							"Going to retry MySQL operation: retry_count = ", retry_count);
						elem->due_time = now + (g_retry_init_delay << retry_count);
#ifdef POSEIDON_CXX11
						std::rethrow_exception(except);
#else
						boost::rethrow_exception(except);
#endif
					}

					LOG_POSEIDON_ERROR("Max retry count exceeded.");
					dump_sql_to_file(query, err_code, err_msg);
					elem->operation->set_exception(except);
				} else {
					elem->operation->set_success();
				}

				const Mutex::UniqueLock lock(m_mutex);
				m_queue.pop_front();
			}
		}

		void dump_sql_to_file(const std::string &query, long err_code, const std::string &err_msg){
			PROFILE_ME;

			if(g_dump_dir.empty()){
				LOG_POSEIDON_WARNING("MySQL dump is disabled.");
				return;
			}

			const AUTO(local_now, get_local_time());
			const AUTO(dt, break_down_time(local_now));
			char temp[256];
			unsigned len = (unsigned)std::sprintf(temp, "%04u-%02u-%02u_%05u.log", dt.yr, dt.mon, dt.day, (unsigned)::getpid());
			std::string dump_path;
			dump_path.assign(g_dump_dir);
			dump_path.push_back('/');
			dump_path.append(temp, len);

			LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "Creating SQL dump file: ", dump_path);
			UniqueFile dump_file;
			if(!dump_file.reset(::open(dump_path.c_str(), O_WRONLY | O_APPEND | O_CREAT, 0644))){
				const int saved_errno = errno;
				LOG_POSEIDON_FATAL("Error creating SQL dump file: dump_path = ", dump_path,
					", errno = ", saved_errno, ", desc = ", get_error_desc(saved_errno));
				std::abort();
			}

			LOG_POSEIDON_INFO("Writing MySQL dump...");
			Buffer_ostream os;
			len = format_time(temp, sizeof(temp), local_now, false);
			os <<"-- " <<temp <<": err_code = " <<err_code <<", err_msg = " <<err_msg <<std::endl;
			os <<query <<";" <<std::endl;
			os <<std::endl;
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
		}

	public:
		void start(){
			const Mutex::UniqueLock lock(m_mutex);
			Thread(boost::bind(&MySqlThread::thread_proc, this), " D  ").swap(m_thread);
			atomic_store(m_running, true, ATOMIC_RELEASE);
			atomic_store(m_alive, true, ATOMIC_RELEASE);
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
				bool alive;
				{
					const Mutex::UniqueLock lock(m_mutex);
					pending_objects = m_queue.size();
					if(pending_objects == 0){
						break;
					}
					m_queue.front().operation->generate_sql(current_sql);
					alive = atomic_load(m_alive, ATOMIC_CONSUME);
					atomic_store(m_urgent, true, ATOMIC_RELEASE);
					m_new_operation.signal();
				}
				LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO,
					"Waiting for SQL queries to complete: pending_objects = ", pending_objects, ", current_sql = ", current_sql);

				if(!alive){
					LOG_POSEIDON_ERROR("MySQL thread seems dead before the queue is emptied. Trying to recover...");
					try {
						if(m_thread.joinable()){
							m_thread.join();
						}
						LOG_POSEIDON_WARNING("Recreating MySQL thread: pending_objects = ", pending_objects);
						start();
					} catch(MySql::Exception &e){
						LOG_POSEIDON_WARNING("MySql::Exception thrown: code = ", e.get_code(), ", what = ", e.what());
					} catch(std::exception &e){
						LOG_POSEIDON_WARNING("std::exception thrown: what = ", e.what());
					} catch(...){
						LOG_POSEIDON_WARNING("Unknown exception thrown");
					}
					alive = atomic_load(m_alive, ATOMIC_CONSUME);
					if(!alive){
						LOG_POSEIDON_ERROR("We cannot recover from this situation. Continue with no regard to data loss.");
						break;
					}
					stop();
				}

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

			AUTO(due_time, get_fast_mono_clock());
			// 有紧急操作时无视写入延迟，这个逻辑不在这里处理。
			due_time += g_save_delay;

			const Mutex::UniqueLock lock(m_mutex);
			if(!atomic_load(m_running, ATOMIC_CONSUME)){
				LOG_POSEIDON_ERROR("MySQL thread is being shut down.");
				DEBUG_THROW(Exception, sslit("MySQL thread is being shut down"));
			}
			m_queue.push_back(OperationQueueElement(STD_MOVE(operation), due_time));
			OperationQueueElement *const elem = &m_queue.back();
			if(combinable_object){
				const AUTO(old_write_stamp, combinable_object->get_combined_write_stamp());
				if(!old_write_stamp){
					combinable_object->set_combined_write_stamp(elem);
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
	std::vector<boost::shared_ptr<MySqlThread> > g_threads;

	void submit_operation_by_table(const char *table, boost::shared_ptr<OperationBase> operation, bool urgent){
		PROFILE_ME;

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
					LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_DEBUG,
						"Creating new MySQL thread ", i, " for table ", table);
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
			LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_DEBUG,
				"Picking thread ", index, " for table ", table);
			thread = g_threads.at(index);
			route.thread = thread;
		}
	_use_thread:
		assert(probe);
		assert(thread);
		operation->set_probe(STD_MOVE(probe));
		thread->add_operation(STD_MOVE(operation), urgent);
	}
	void submit_operation_all(boost::shared_ptr<OperationBase> operation, bool urgent){
		PROFILE_ME;

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

	MainConfig::get(g_master_addr, "mysql_server_addr");
	LOG_POSEIDON_DEBUG("MySQL master addr = ", g_master_addr);

	MainConfig::get(g_master_port, "mysql_server_port");
	LOG_POSEIDON_DEBUG("MySQL master port = ", g_master_port);

	MainConfig::get(g_slave_addr, "mysql_slave_addr");
	LOG_POSEIDON_DEBUG("MySQL slave addr = ", g_slave_addr);

	MainConfig::get(g_slave_port, "mysql_slave_port");
	LOG_POSEIDON_DEBUG("MySQL slave port = ", g_slave_port);

	MainConfig::get(g_username, "mysql_username");
	LOG_POSEIDON_DEBUG("MySQL username = ", g_username);

	MainConfig::get(g_password, "mysql_password");
	LOG_POSEIDON_DEBUG("MySQL password = ", g_password);

	MainConfig::get(g_schema, "mysql_schema");
	LOG_POSEIDON_DEBUG("MySQL schema = ", g_schema);

	MainConfig::get(g_use_ssl, "mysql_use_ssl");
	LOG_POSEIDON_DEBUG("MySQL use ssl = ", g_use_ssl);

	MainConfig::get(g_charset, "mysql_charset");
	LOG_POSEIDON_DEBUG("MySQL charset = ", g_charset);

	MainConfig::get(g_dump_dir, "mysql_dump_dir");
	LOG_POSEIDON_DEBUG("MySQL dump dir = ", g_dump_dir);

	MainConfig::get(g_save_delay, "mysql_save_delay");
	LOG_POSEIDON_DEBUG("MySQL save delay = ", g_save_delay);

	MainConfig::get(g_reconn_delay, "mysql_reconn_delay");
	LOG_POSEIDON_DEBUG("MySQL reconnect delay = ", g_reconn_delay);

	MainConfig::get(g_max_retry_count, "mysql_max_retry_count");
	LOG_POSEIDON_DEBUG("MySQL max retry count = ", g_max_retry_count);

	MainConfig::get(g_retry_init_delay, "mysql_retry_init_delay");
	LOG_POSEIDON_DEBUG("MySQL retry init delay = ", g_retry_init_delay);

	MainConfig::get(g_max_thread_count, "mysql_max_thread_count");
	LOG_POSEIDON_DEBUG("MySQL max thread count = ", g_max_thread_count);

	if(!g_dump_dir.empty()){
		const AUTO(placeholder_path, g_dump_dir + "/placeholder");
		LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO,
			"Checking whether MySQL dump directory is writeable: test_path = ", placeholder_path);
		UniqueFile dump_file;
		if(!dump_file.reset(::open(placeholder_path.c_str(), O_WRONLY | O_TRUNC | O_CREAT, 0644))){
			const int err_code = errno;
			LOG_POSEIDON_FATAL("Could not create placeholder file: placeholder_path = ", placeholder_path, ", errno = ", err_code);
			std::abort();
		}
	}

	g_threads.resize(std::max<std::size_t>(g_max_thread_count, 1));

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
	AUTO(addr, &g_master_addr);
	AUTO(port, &g_master_port);
	if(from_slave){
		if(!g_slave_addr.empty()){
			addr = &g_slave_addr;
		}
		if(g_slave_port != 0){
			port = &g_slave_port;
		}
	}
	return MySql::Connection::create(*addr, *port, g_username, g_password, g_schema, g_use_ssl, g_charset);
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

boost::shared_ptr<const JobPromise> MySqlDaemon::enqueue_for_saving(
	boost::shared_ptr<const MySql::ObjectBase> object, bool to_replace, bool urgent)
{
	AUTO(promise, boost::make_shared<JobPromise>());
	const char *const table_name = object->get_table_name();
	AUTO(operation, boost::make_shared<SaveOperation>(promise, STD_MOVE(object), to_replace));
	submit_operation_by_table(table_name, STD_MOVE_IDN(operation), urgent);
	return STD_MOVE_IDN(promise);
}
boost::shared_ptr<const JobPromise> MySqlDaemon::enqueue_for_loading(
	boost::shared_ptr<MySql::ObjectBase> object, std::string query)
{
	AUTO(promise, boost::make_shared<JobPromise>());
	const char *const table_name = object->get_table_name();
	AUTO(operation, boost::make_shared<LoadOperation>(promise, STD_MOVE(object), STD_MOVE(query)));
	submit_operation_by_table(table_name, STD_MOVE_IDN(operation), true);
	return STD_MOVE_IDN(promise);
}
boost::shared_ptr<const JobPromise> MySqlDaemon::enqueue_for_deleting(
	const char *table_hint, std::string query)
{
	AUTO(promise, boost::make_shared<JobPromise>());
	const char *const table_name = table_hint;
	AUTO(operation, boost::make_shared<DeleteOperation>(promise, table_hint, STD_MOVE(query)));
	submit_operation_by_table(table_name, STD_MOVE_IDN(operation), true);
	return STD_MOVE_IDN(promise);
}
boost::shared_ptr<const JobPromise> MySqlDaemon::enqueue_for_batch_loading(
	MySqlDaemon::ObjectFactory factory, const char *table_hint, std::string query)
{
	AUTO(promise, boost::make_shared<JobPromise>());
	const char *const table_name = table_hint;
	AUTO(operation, boost::make_shared<BatchLoadOperation>(promise, STD_MOVE(factory), table_hint, STD_MOVE(query)));
	submit_operation_by_table(table_name, STD_MOVE_IDN(operation), true);
	return STD_MOVE_IDN(promise);
}

boost::shared_ptr<const JobPromise> MySqlDaemon::enqueue_for_waiting_for_all_async_operations(){
	AUTO(promise, boost::make_shared<JobPromise>());
	AUTO(operation, boost::make_shared<WaitOperation>(promise));
	submit_operation_all(STD_MOVE_IDN(operation), true);
	return STD_MOVE_IDN(promise);
}

}
