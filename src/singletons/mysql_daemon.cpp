// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "mysql_daemon.hpp"
#include "main_config.hpp"
#include <boost/container/vector.hpp>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <mysql/mysqld_error.h>
#include <mysql/errmsg.h>
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

namespace Poseidon {

namespace {
	std::string     g_server_addr       = "localhost";
	unsigned        g_server_port       = 3306;
	std::string     g_slave_addr        = VAL_INIT;
	unsigned        g_slave_port        = 0;
	std::string     g_username          = "root";
	std::string     g_password          = "root";
	std::string     g_schema            = "poseidon";
	bool            g_use_ssl           = false;
	std::string     g_charset           = "utf8";

	std::string     g_dump_dir          = VAL_INIT;
	std::size_t     g_max_threads       = 3;
	boost::uint64_t g_save_delay        = 5000;
	boost::uint64_t g_reconn_delay      = 10000;
	std::size_t     g_max_retry_count   = 3;
	boost::uint64_t g_retry_init_delay  = 1000;

	// 对于日志文件的写操作应当互斥。
	Mutex g_dump_mutex;

	// 数据库线程操作。
	class OperationBase : NONCOPYABLE {
	private:
		const boost::shared_ptr<JobPromise> m_promise;

	public:
		explicit OperationBase(boost::shared_ptr<JobPromise> promise)
			: m_promise(STD_MOVE(promise))
		{
		}
		virtual ~OperationBase(){
		}

	public:
		const boost::shared_ptr<JobPromise> &get_promise() const {
			return m_promise;
		}

		virtual bool should_use_slave() const = 0;
		virtual boost::shared_ptr<const MySql::ObjectBase> get_combinable_object() const = 0;
		virtual const char *get_table_name() const = 0;
		virtual void execute(std::string &query, const boost::shared_ptr<MySql::Connection> &conn) const = 0;
	};

	class SaveOperation : public OperationBase {
	private:
		const boost::shared_ptr<const MySql::ObjectBase> m_object;
		const bool m_to_replace;

	public:
		SaveOperation(boost::shared_ptr<JobPromise> promise,
			boost::shared_ptr<const MySql::ObjectBase> object, bool to_replace)
			: OperationBase(STD_MOVE(promise))
			, m_object(STD_MOVE(object)), m_to_replace(to_replace)
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
		void execute(std::string &query, const boost::shared_ptr<MySql::Connection> &conn) const OVERRIDE {
			PROFILE_ME;

			m_object->generate_sql(query, m_to_replace);
			LOG_POSEIDON_DEBUG("Executing SQL in ", m_object->get_table_name(), ": query = ", query);
			conn->execute_sql(query);
		}
	};

	class LoadOperation : public OperationBase {
	private:
		const boost::shared_ptr<MySql::ObjectBase> m_object;
		const std::string m_query;

	public:
		LoadOperation(boost::shared_ptr<JobPromise> promise,
			boost::shared_ptr<MySql::ObjectBase> object, std::string query)
			: OperationBase(STD_MOVE(promise))
			, m_object(STD_MOVE(object)), m_query(STD_MOVE(query))
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
		void execute(std::string &query, const boost::shared_ptr<MySql::Connection> &conn) const OVERRIDE {
			PROFILE_ME;

			query = m_query;
			LOG_POSEIDON_INFO("MySQL load: table = ", m_object->get_table_name(), ", query = ", query);
			conn->execute_sql(query);

			if(!conn->fetch_row()){
				DEBUG_THROW(MySql::Exception, 99999, sslit("No rows returned"));
			}
			m_object->disable_auto_saving();
			m_object->fetch(conn);
			m_object->enable_auto_saving();
		}
	};

	class BatchLoadOperation : public OperationBase {
	private:
		boost::shared_ptr<std::deque<boost::shared_ptr<MySql::ObjectBase> > > m_container;
		boost::shared_ptr<MySql::ObjectBase> (*const m_factory)();
		const char *const m_table_hint;
		const std::string m_query;

	public:
		BatchLoadOperation(boost::shared_ptr<JobPromise> promise,
			boost::shared_ptr<std::deque<boost::shared_ptr<MySql::ObjectBase> > > container,
			boost::shared_ptr<MySql::ObjectBase> (*factory)(), const char *table_hint, std::string query)
			: OperationBase(STD_MOVE(promise))
			, m_container(STD_MOVE(container)), m_factory(factory), m_table_hint(table_hint), m_query(STD_MOVE(query))
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
		void execute(std::string &query, const boost::shared_ptr<MySql::Connection> &conn) const OVERRIDE {
			PROFILE_ME;

			query = m_query;
			LOG_POSEIDON_INFO("MySQL batch load: table_hint = ", m_table_hint, ", query = ", query);
			conn->execute_sql(query);

			if(m_factory && m_container){
				while(conn->fetch_row()){
					AUTO(object, (*m_factory)());
					object->fetch(conn);
					object->enable_auto_saving();
					m_container->push_back(STD_MOVE(object));
				}
			} else {
				LOG_POSEIDON_DEBUG("Result discarded.");
			}
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

		class WorkingTimeAccumulator : NONCOPYABLE {
		private:
			MySqlThread *const m_owner;
			const char *const m_table;

		public:
			WorkingTimeAccumulator(MySqlThread *owner, const char *table)
				: m_owner(owner), m_table(table)
			{
				m_owner->accumulate_time_for_table("");
			}
			~WorkingTimeAccumulator(){
				m_owner->accumulate_time_for_table(m_table);
			}
		};

		struct TableNameComparator {
			bool operator()(const char *lhs, const char *rhs) const {
				return std::strcmp(lhs, rhs) < 0;
			}
		};

	private:
		Thread m_thread;
		volatile bool m_running;

		mutable Mutex m_mutex;
		mutable ConditionVariable m_new_operation;
		volatile bool m_urgent; // 无视延迟写入，一次性处理队列中所有操作。
		std::deque<OperationQueueElement> m_queue;

		// 性能统计。
		mutable Mutex m_profile_mutex;
		double m_profile_flushed_time;
		std::map<const char *, unsigned long long, TableNameComparator> m_profile;

	public:
		MySqlThread()
			: m_running(false)
			, m_urgent(false)
			, m_profile_flushed_time(0)
		{
		}

	private:
		void accumulate_time_for_table(const char *table) NOEXCEPT {
			const AUTO(now, get_hi_res_mono_clock());

			const Mutex::UniqueLock lock(m_profile_mutex);
			try {
				m_profile[table] += (now - m_profile_flushed_time) * 1e6;
			} catch(...){
			}
			m_profile_flushed_time = now;
		}

		void sleep_for_reconnection(){
			::timespec req;
			req.tv_sec = (::time_t)(g_reconn_delay / 1000);
			req.tv_nsec = (long)(g_reconn_delay % 1000) * 1000 * 1000;
			::nanosleep(&req, NULLPTR);
		}

		void thread_proc(){
			PROFILE_ME;
			LOG_POSEIDON_INFO("MySQL thread started.");

			m_profile_flushed_time = get_hi_res_mono_clock();

			const MySql::ThreadContext thread_context;
			boost::shared_ptr<MySql::Connection> master_conn, slave_conn;

			for(;;){
				accumulate_time_for_table("");

				while(!master_conn){
					LOG_POSEIDON_INFO("Connecting to MySQL master server...");
					try {
						master_conn = MySqlDaemon::create_connection(false);
						LOG_POSEIDON_INFO("Successfully connected to MySQL master server.");
					} catch(std::exception &e){
						LOG_POSEIDON_ERROR("std::exception thrown: what = ", e.what());
						sleep_for_reconnection();
					}
				}
				while(!slave_conn){
					LOG_POSEIDON_INFO("Connecting to MySQL slave server...");
					try {
						slave_conn = MySqlDaemon::create_connection(true);
						LOG_POSEIDON_INFO("Successfully connected to MySQL slave server.");
					} catch(std::exception &e){
						LOG_POSEIDON_ERROR("std::exception thrown: what = ", e.what());
						sleep_for_reconnection();
					}
				}

				bool should_retry = true;
				try {
					really_pump_operations(master_conn, slave_conn);
					should_retry = false;
				} catch(MySql::Exception &e){
					LOG_POSEIDON_WARNING("MySql::Exception thrown: code = ", e.code(), ", what = ", e.what());
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

			LOG_POSEIDON_INFO("MySQL thread stopped.");
		}

		void really_pump_operations(const boost::shared_ptr<MySql::Connection> &master_conn,
			const boost::shared_ptr<MySql::Connection> &slave_conn)
		{
			PROFILE_ME;

			const AUTO(now, get_fast_mono_clock());

			for(;;){
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
					boost::exception_ptr except;

					long err_code = 0;
					char message[4096];
					std::size_t message_len = 0;

					std::string query;

					try {
						const WorkingTimeAccumulator profiler(this, elem->operation->get_table_name());
						elem->operation->execute(query, conn);
					} catch(MySql::Exception &e){
						LOG_POSEIDON_WARNING("MySql::Exception thrown: code = ", e.code(), ", what = ", e.what());
						// except = boost::current_exception();
						except = boost::copy_exception(e);

						err_code = e.code();
						message_len = std::min(std::strlen(e.what()), sizeof(message));
						std::memcpy(message, e.what(), message_len);
					} catch(std::exception &e){
						LOG_POSEIDON_WARNING("std::exception thrown: what = ", e.what());
						// except = boost::current_exception();
						except = boost::copy_exception(std::runtime_error(e.what()));

						err_code = 99999;
						message_len = std::min(std::strlen(e.what()), sizeof(message));
						std::memcpy(message, e.what(), message_len);
					} catch(...){
						LOG_POSEIDON_WARNING("Unknown exception thrown");
						except = boost::current_exception();

						err_code = 99999;
						message_len = 17;
						std::memcpy(message, "Unknown exception", 17);
					}
					conn->discard_result();

					const AUTO(promise, elem->operation->get_promise());
					if(except){
						const AUTO(retry_count, ++elem->retry_count);
						if(retry_count < g_max_retry_count){
							LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO,
								"Going to retry MySQL operation: retry_count = ", retry_count);
							elem->due_time = now + (g_retry_init_delay << retry_count);
							boost::rethrow_exception(except);
						}

						LOG_POSEIDON_ERROR("Max retry count exceeded.");
						dump_sql(query, err_code, message, message_len);
						promise->set_exception(except);
					} else {
						promise->set_success();
					}
				}

				const Mutex::UniqueLock lock(m_mutex);
				m_queue.pop_front();
			}
		}

		void dump_sql(const std::string &query, long err_code, const char *message, std::size_t message_len){
			PROFILE_ME;

			if(g_dump_dir.empty()){
				LOG_POSEIDON_WARNING("MySQL dump is disabled.");
				return;
			}

			const AUTO(local_now, get_local_time());
			const AUTO(dt, break_down_time(local_now));
			char temp[256];
			unsigned len = (unsigned)std::sprintf(temp, "%04u-%02u-%02u %05u", dt.yr, dt.mon, dt.day, (unsigned)::getpid());
			std::string dump_path;
			dump_path.assign(g_dump_dir);
			dump_path.push_back('/');
			dump_path.append(temp, len);
			dump_path.append(".log");

			LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "Creating SQL dump file: ", dump_path);
			UniqueFile dump_file;
			if(!dump_file.reset(::open(dump_path.c_str(), O_WRONLY | O_APPEND | O_CREAT, 0644))){
				const int err_code = errno;
				LOG_POSEIDON_FATAL("Error creating SQL dump file: dump_path = ", dump_path,
					", errno = ", err_code, ", desc = ", get_error_desc(err_code));
				std::abort();
			}

			LOG_POSEIDON_INFO("Writing MySQL dump...");
			std::string dump;
			dump.reserve(1024);
			dump.append("-- Time = ");
			len = format_time(temp, sizeof(temp), local_now, false);
			dump.append(temp, len);
			dump.append(", Error code = ");
			len = (unsigned)std::sprintf(temp, "%ld", err_code);
			dump.append(temp, len);
			dump.append(", Description = ");
			dump.append(message, message_len);
			dump.append("\n");
			dump.append(query);
			dump.append(";\n\n");

			const Mutex::UniqueLock lock(g_dump_mutex);
			std::size_t total = 0;
			do {
				::ssize_t written = ::write(dump_file.get(), dump.data() + total, dump.size() - total);
				if(written <= 0){
					break;
				}
				total += static_cast<std::size_t>(written);
			} while(total < dump.size());
		}

	public:
		void start(){
			Thread(boost::bind(&MySqlThread::thread_proc, this), " D  ").swap(m_thread);
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

		void get_profile(std::vector<std::pair<const char *, unsigned long long> > &ret) const {
			const Mutex::UniqueLock lock(m_profile_mutex);
			ret.reserve(ret.size() + m_profile.size());
			for(AUTO(it, m_profile.begin()); it != m_profile.end(); ++it){
				ret.push_back(std::make_pair(it->first, it->second));
			}
		}

		void wait_till_idle(){
			for(;;){
				std::size_t pending_objects;
				{
					const Mutex::UniqueLock lock(m_mutex);
					pending_objects = m_queue.size();
					if(pending_objects == 0){
						break;
					}
					atomic_store(m_urgent, true, ATOMIC_RELEASE);
					m_new_operation.signal();
				}
				LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "There are ", pending_objects, " object(s) in my queue.");

				::timespec req;
				req.tv_sec = 0;
				req.tv_nsec = 500 * 1000 * 1000;
				::nanosleep(&req, NULLPTR);
			}
		}

		void add_operation(boost::shared_ptr<OperationBase> operation, bool urgent){
			PROFILE_ME;

			if(!atomic_load(m_running, ATOMIC_CONSUME)){
				LOG_POSEIDON_ERROR("MySQL thread is being shut down.");
				DEBUG_THROW(Exception, sslit("MySQL thread is being shut down"));
			}

			const AUTO(combinable_object, operation->get_combinable_object());

			AUTO(due_time, get_fast_mono_clock());
			// 有紧急操作时无视写入延迟，这个逻辑不在这里处理。
			// if(combinable_object && !urgent) // 这个看似合理但是实际是错的。
			if(combinable_object || urgent){ // 确保紧急操作排在其他所有操作之后。
				due_time += g_save_delay;
			}

			const Mutex::UniqueLock lock(m_mutex);
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
	std::vector<boost::shared_ptr<MySqlThread>> g_threads;

	void submit_operation(const char *table, boost::shared_ptr<OperationBase> operation, bool urgent){
		if(g_threads.empty()){
			DEBUG_THROW(Exception, sslit("No MySQL thread is running"));
		}

		// http://www.isthe.com/chongo/tech/comp/fnv/
		std::size_t hash;
		if(sizeof(std::size_t) < 8){
			hash = 2166136261u;
			const char *p = table;
			while(*p){
				hash ^= static_cast<unsigned char>(*p);
				hash *= 16777619u;
				++p;
			}
		} else {
			hash = 14695981039346656037u;
			const char *p = table;
			while(*p){
				hash ^= static_cast<unsigned char>(*p);
				hash *= 1099511628211u;
				++p;
			}
		}
		const AUTO(thread_index, hash % g_threads.size());
		LOG_POSEIDON_DEBUG("Assigning MySQL table `", table, "` to thread ", thread_index);

		g_threads.at(thread_index)->add_operation(STD_MOVE(operation), urgent);
	}
}

void MySqlDaemon::start(){
	if(atomic_exchange(g_running, true, ATOMIC_ACQ_REL) != false){
		LOG_POSEIDON_FATAL("Only one daemon is allowed at the same time.");
		std::abort();
	}
	LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "Starting MySQL daemon...");

	MainConfig::get(g_server_addr, "mysql_server_addr");
	LOG_POSEIDON_DEBUG("MySQL server addr = ", g_server_addr);

	MainConfig::get(g_server_port, "mysql_server_port");
	LOG_POSEIDON_DEBUG("MySQL server port = ", g_server_port);

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

	MainConfig::get(g_max_threads, "mysql_max_threads");
	LOG_POSEIDON_DEBUG("MySQL max threads = ", g_max_threads);

	MainConfig::get(g_save_delay, "mysql_save_delay");
	LOG_POSEIDON_DEBUG("MySQL save delay = ", g_save_delay);

	MainConfig::get(g_reconn_delay, "mysql_reconn_delay");
	LOG_POSEIDON_DEBUG("MySQL reconnect delay = ", g_reconn_delay);

	MainConfig::get(g_max_retry_count, "mysql_max_retry_count");
	LOG_POSEIDON_DEBUG("MySQL max retry count = ", g_max_retry_count);

	MainConfig::get(g_retry_init_delay, "mysql_retry_init_delay");
	LOG_POSEIDON_DEBUG("MySQL retry init delay = ", g_retry_init_delay);

	const AUTO(thread_count, std::max<std::size_t>(g_max_threads, 1));
	g_threads.clear();
	g_threads.reserve(thread_count);
	for(std::size_t i = 0; i < thread_count; ++i){
		LOG_POSEIDON_INFO("Creating MySQL thread ", i);
		AUTO(thread, boost::make_shared<MySqlThread>());
		thread->start();
		g_threads.push_back(STD_MOVE(thread));
	}

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

	LOG_POSEIDON_INFO("MySQL daemon started.");
}
void MySqlDaemon::stop(){
	if(atomic_exchange(g_running, false, ATOMIC_ACQ_REL) == false){
		return;
	}
	LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "Stopping MySQL daemon...");

	const AUTO(threads, STD_MOVE_IDN(g_threads));
	g_threads.clear();
	for(std::size_t i = 0; i < threads.size(); ++i){
		LOG_POSEIDON_INFO("Stopping MySQL thread ", i);
		threads.at(i)->stop();
	}
	for(std::size_t i = 0; i < threads.size(); ++i){
		LOG_POSEIDON_INFO("Waiting for MySQL thread ", i, " to terminate...");
		threads.at(i)->safe_join();
	}

	LOG_POSEIDON_INFO("MySQL daemon stopped.");
}

boost::shared_ptr<MySql::Connection> MySqlDaemon::create_connection(bool from_slave){
	AUTO(addr, &g_server_addr);
	AUTO(port, &g_server_port);
	if(from_slave){
		if(!g_slave_addr.empty()){
			addr = &g_slave_addr;
		}
		if(g_server_port != 0){
			port = &g_slave_port;
		}
	}
	return MySql::Connection::create(*addr, *port, g_username, g_password, g_schema, g_use_ssl, g_charset);
}

std::vector<MySqlDaemon::SnapshotElement> MySqlDaemon::snapshot(){
	std::vector<SnapshotElement> ret;
	ret.reserve(100);

	std::vector<std::pair<const char *, unsigned long long> > thread_profile;
	for(std::size_t i = 0; i < g_threads.size(); ++i){
		g_threads.at(i)->get_profile(thread_profile);
		ret.reserve(ret.size() + thread_profile.size());
		for(AUTO(it, thread_profile.begin()); it != thread_profile.end(); ++it){
			ret.push_back(VAL_INIT);
			AUTO_REF(elem, ret.back());
			elem.thread = i;
			elem.table = it->first;
			elem.ns_total = it->second;
		}
	}

	return ret;
}

void MySqlDaemon::wait_for_all_async_operations(){
	for(std::size_t i = 0; i < g_threads.size(); ++i){
		g_threads.at(i)->wait_till_idle();
	}
}

boost::shared_ptr<const JobPromise> MySqlDaemon::enqueue_for_saving(
	boost::shared_ptr<const MySql::ObjectBase> object, bool to_replace, bool urgent)
{
	AUTO(promise, boost::make_shared<JobPromise>());
	const char *const table_name = object->get_table_name();
	submit_operation(table_name,
		boost::make_shared<SaveOperation>(promise, STD_MOVE(object), to_replace),
		urgent);
	return STD_MOVE_IDN(promise);
}
boost::shared_ptr<const JobPromise> MySqlDaemon::enqueue_for_loading(
	boost::shared_ptr<MySql::ObjectBase> object, std::string query)
{
	AUTO(promise, boost::make_shared<JobPromise>());
	const char *const table_name = object->get_table_name();
	submit_operation(table_name,
		boost::make_shared<LoadOperation>(promise, STD_MOVE(object), STD_MOVE(query)),
		true);
	return STD_MOVE_IDN(promise);
}
boost::shared_ptr<const JobPromise> MySqlDaemon::enqueue_for_batch_loading(
	boost::shared_ptr<std::deque<boost::shared_ptr<MySql::ObjectBase> > > sink,
	boost::shared_ptr<MySql::ObjectBase> (*factory)(), const char *table_hint, std::string query)
{
	AUTO(promise, boost::make_shared<JobPromise>());
	const char *const table_name = table_hint;
	submit_operation(table_name,
		boost::make_shared<BatchLoadOperation>(promise, STD_MOVE(sink), factory, table_hint, STD_MOVE(query)),
		true);
	return STD_MOVE_IDN(promise);
}

}
