// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2016, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "mongodb_daemon.hpp"
#include "main_config.hpp"
#include <boost/container/flat_map.hpp>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#pragma GCC push_options
#pragma GCC diagnostic ignored "-Wsign-conversion"
#include <bson.h>
#include <mongoc.h>
#pragma GCC pop_options
#include "../mongodb/object_base.hpp"
#include "../mongodb/exception.hpp"
#include "../mongodb/connection.hpp"
#include "../mongodb/oid.hpp"
#include "../mongodb/bson_builder.hpp"
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
	std::string     g_auth_database     = "admin";
	std::string     g_database          = "poseidon";
	bool            g_use_ssl           = false;

	std::string     g_dump_dir          = VAL_INIT;
	boost::uint64_t g_save_delay        = 5000;
	boost::uint64_t g_reconn_delay      = 10000;
	std::size_t     g_max_retry_count   = 3;
	boost::uint64_t g_retry_init_delay  = 1000;

	// 对于日志文件的写操作应当互斥。
	Mutex g_dump_mutex;

	typedef MongoDbDaemon::ObjectFactory ObjectFactory;

	// 数据库线程操作。
	class OperationBase : NONCOPYABLE {
	public:
		virtual ~OperationBase(){
		}

	public:
		virtual bool should_use_slave() const = 0;
		virtual boost::shared_ptr<const MongoDb::ObjectBase> get_combinable_object() const = 0;
		virtual const char *get_collection_name() const = 0;
		virtual MongoDb::BsonBuilder generate_query() const = 0;
		virtual void execute(const boost::shared_ptr<MongoDb::Connection> &conn, const MongoDb::BsonBuilder &query) const = 0;
		virtual void set_success() const = 0;
		virtual void set_exception(boost::exception_ptr ep) const = 0;
	};

	class SaveOperation : public OperationBase {
	private:
		const boost::shared_ptr<JobPromise> m_promise;
		const boost::shared_ptr<const MongoDb::ObjectBase> m_object;
		const bool m_to_replace;

	public:
		SaveOperation(boost::shared_ptr<JobPromise> promise,
			boost::shared_ptr<const MongoDb::ObjectBase> object, bool to_replace)
			: m_promise(STD_MOVE(promise)), m_object(STD_MOVE(object)), m_to_replace(to_replace)
		{
		}

	protected:
		bool should_use_slave() const {
			return false;
		}
		boost::shared_ptr<const MongoDb::ObjectBase> get_combinable_object() const OVERRIDE {
			return m_object;
		}
		const char *get_collection_name() const OVERRIDE {
			return m_object->get_collection_name();
		}
		MongoDb::BsonBuilder generate_query() const OVERRIDE {
			MongoDb::BsonBuilder doc;
			m_object->generate_document(doc);
			return doc;
		}
		void execute(const boost::shared_ptr<MongoDb::Connection> &conn, const MongoDb::BsonBuilder &query) const OVERRIDE {
			PROFILE_ME;

			if(m_to_replace){
				conn->execute_replace(get_collection_name(), query);
			} else {
				conn->execute_insert(get_collection_name(), query, true);
			}
		}
		void set_success() const OVERRIDE {
			m_promise->set_success();
		}
		void set_exception(boost::exception_ptr ep) const OVERRIDE {
			m_promise->set_exception(STD_MOVE(ep));
		}
	};

	class LoadOperation : public OperationBase {
	private:
		const boost::shared_ptr<JobPromise> m_promise;
		const boost::shared_ptr<MongoDb::ObjectBase> m_object;
		const MongoDb::BsonBuilder m_query;

	public:
		LoadOperation(boost::shared_ptr<JobPromise> promise,
			boost::shared_ptr<MongoDb::ObjectBase> object, MongoDb::BsonBuilder query)
			: m_promise(STD_MOVE(promise)), m_object(STD_MOVE(object)), m_query(STD_MOVE(query))
		{
		}

	protected:
		bool should_use_slave() const {
			return true;
		}
		boost::shared_ptr<const MongoDb::ObjectBase> get_combinable_object() const OVERRIDE {
			return VAL_INIT; // 不能合并。
		}
		const char *get_collection_name() const OVERRIDE {
			return m_object->get_collection_name();
		}
		MongoDb::BsonBuilder generate_query() const OVERRIDE {
			return m_query;
		}
		void execute(const boost::shared_ptr<MongoDb::Connection> &conn, const MongoDb::BsonBuilder &query) const OVERRIDE {
			PROFILE_ME;

			conn->execute_query(get_collection_name(), query, 0, 1);
			if(!conn->fetch_next()){
				DEBUG_THROW(MongoDb::Exception, SharedNts::view(get_collection_name()), MONGOC_ERROR_QUERY_FAILURE, sslit("No rows returned"));
			}
			m_object->fetch(conn);
		}
		void set_success() const OVERRIDE {
			m_promise->set_success();
		}
		void set_exception(boost::exception_ptr ep) const OVERRIDE {
			m_promise->set_exception(STD_MOVE(ep));
		}
	};

	class DeleteOperation : public OperationBase {
	private:
		const boost::shared_ptr<JobPromise> m_promise;
		const char *const m_collection;
		const MongoDb::BsonBuilder m_query;
		const bool m_delete_all;

	public:
		DeleteOperation(boost::shared_ptr<JobPromise> promise,
			const char *collection, MongoDb::BsonBuilder query, bool delete_all)
			: m_promise(STD_MOVE(promise)), m_collection(collection), m_query(STD_MOVE(query)), m_delete_all(delete_all)
		{
		}

	protected:
		bool should_use_slave() const {
			return false;
		}
		boost::shared_ptr<const MongoDb::ObjectBase> get_combinable_object() const OVERRIDE {
			return VAL_INIT; // 不能合并。
		}
		const char *get_collection_name() const OVERRIDE {
			return m_collection;
		}
		MongoDb::BsonBuilder generate_query() const OVERRIDE {
			return m_query;
		}
		void execute(const boost::shared_ptr<MongoDb::Connection> &conn, const MongoDb::BsonBuilder &query) const OVERRIDE {
			PROFILE_ME;

			conn->execute_delete(get_collection_name(), query, m_delete_all);
		}
		void set_success() const OVERRIDE {
			m_promise->set_success();
		}
		void set_exception(boost::exception_ptr ep) const OVERRIDE {
			m_promise->set_exception(STD_MOVE(ep));
		}
	};

	class BatchLoadOperation : public OperationBase {
	private:
		const boost::shared_ptr<JobPromise> m_promise;
		const ObjectFactory m_factory;
		const char *const m_collection;
		const MongoDb::BsonBuilder m_query;
		const boost::uint32_t m_begin;
		const boost::uint32_t m_limit;

	public:
		BatchLoadOperation(boost::shared_ptr<JobPromise> promise,
			ObjectFactory factory, const char *collection, MongoDb::BsonBuilder query, boost::uint32_t begin, boost::uint32_t limit)
			: m_promise(STD_MOVE(promise)), m_factory(STD_MOVE_IDN(factory)), m_collection(collection)
			, m_query(STD_MOVE(query)), m_begin(begin), m_limit(limit)
		{
		}

	protected:
		bool should_use_slave() const {
			return true;
		}
		boost::shared_ptr<const MongoDb::ObjectBase> get_combinable_object() const OVERRIDE {
			return VAL_INIT; // 不能合并。
		}
		const char *get_collection_name() const OVERRIDE {
			return m_collection;
		}
		MongoDb::BsonBuilder generate_query() const OVERRIDE {
			return m_query;
		}
		void execute(const boost::shared_ptr<MongoDb::Connection> &conn, const MongoDb::BsonBuilder &query) const OVERRIDE {
			PROFILE_ME;

			conn->execute_query(get_collection_name(), query, m_begin, m_limit);
			if(m_factory){
				while(conn->fetch_next()){
					m_factory(conn);
				}
			} else {
				LOG_POSEIDON_DEBUG("Result discarded.");
			}
		}
		void set_success() const OVERRIDE {
			m_promise->set_success();
		}
		void set_exception(boost::exception_ptr ep) const OVERRIDE {
			m_promise->set_exception(STD_MOVE(ep));
		}
	};

	class WaitOperation : public OperationBase {
	private:
		const boost::shared_ptr<JobPromise> m_promise;
		const boost::shared_ptr<volatile std::size_t> m_counter;

	public:
		WaitOperation(boost::shared_ptr<JobPromise> promise,
			boost::shared_ptr<volatile std::size_t> counter)
			: m_promise(STD_MOVE(promise)), m_counter(STD_MOVE(counter))
		{
		}

	protected:
		bool should_use_slave() const {
			return false;
		}
		boost::shared_ptr<const MongoDb::ObjectBase> get_combinable_object() const OVERRIDE {
			return VAL_INIT; // 不能合并。
		}
		const char *get_collection_name() const OVERRIDE {
			return "";
		}
		MongoDb::BsonBuilder generate_query() const OVERRIDE {
			return MongoDb::bson_scalar_signed(sslit("ping"), 1);
		}
		void execute(const boost::shared_ptr<MongoDb::Connection> &conn, const MongoDb::BsonBuilder &query) const OVERRIDE {
			PROFILE_ME;

			conn->execute_command(get_collection_name(), query, 0, 1);
		}
		void set_success() const OVERRIDE {
			if(atomic_sub(*m_counter, 1, ATOMIC_RELAXED) != 0){
				return;
			}
			m_promise->set_success();
		}
		void set_exception(boost::exception_ptr ep) const OVERRIDE {
			if(atomic_sub(*m_counter, 1, ATOMIC_RELAXED) != 0){
				return;
			}
			m_promise->set_exception(STD_MOVE(ep));
		}
	};

	class MongoDbThread : NONCOPYABLE {
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
		std::deque<OperationQueueElement> m_queue;

	public:
		MongoDbThread()
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
			LOG_POSEIDON_INFO("MongoDB thread started.");

			boost::shared_ptr<MongoDb::Connection> master_conn, slave_conn;

			for(;;){
				while(!master_conn){
					LOG_POSEIDON_INFO("Connecting to MongoDB master server...");
					try {
						master_conn = MongoDbDaemon::create_connection(false);
						LOG_POSEIDON_INFO("Successfully connected to MongoDB master server.");
					} catch(std::exception &e){
						LOG_POSEIDON_ERROR("std::exception thrown: what = ", e.what());
						sleep_for_reconnection();
					}
				}
				while(!slave_conn){
					LOG_POSEIDON_INFO("Connecting to MongoDB slave server...");
					try {
						slave_conn = MongoDbDaemon::create_connection(true);
						LOG_POSEIDON_INFO("Successfully connected to MongoDB slave server.");
					} catch(std::exception &e){
						LOG_POSEIDON_ERROR("std::exception thrown: what = ", e.what());
						sleep_for_reconnection();
					}
				}

				bool should_retry = true;
				try {
					really_pump_operations(master_conn, slave_conn);
					should_retry = false;
				} catch(MongoDb::Exception &e){
					LOG_POSEIDON_WARNING("MongoDb::Exception thrown: database = ", e.get_database(),
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
			LOG_POSEIDON_INFO("MongoDB thread stopped.");
		}

		void really_pump_operations(const boost::shared_ptr<MongoDb::Connection> &master_conn,
			const boost::shared_ptr<MongoDb::Connection> &slave_conn)
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

				const AUTO_REF(operation, elem->operation);
				boost::exception_ptr except;

				MongoDb::BsonBuilder query;
				boost::uint32_t err_code = 0;
				char message[4096];
				std::size_t message_len = 0;

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
					query = operation->generate_query();
					try {
						LOG_POSEIDON_DEBUG("Executing SQL: collection_name = ", operation->get_collection_name(), ", query = ", query);
						operation->execute(conn, query);
					} catch(MongoDb::Exception &e){
						LOG_POSEIDON_WARNING("MongoDb::Exception thrown: code = ", e.get_code(), ", what = ", e.what());
						// except = boost::current_exception();
						except = boost::copy_exception(e);

						err_code = e.get_code();
						message_len = std::min(std::strlen(e.what()), sizeof(message));
						std::memcpy(message, e.what(), message_len);
					} catch(std::exception &e){
						LOG_POSEIDON_WARNING("std::exception thrown: what = ", e.what());
						// except = boost::current_exception();
						except = boost::copy_exception(e);

						err_code = MONGOC_ERROR_PROTOCOL_ERROR;
						message_len = std::min(std::strlen(e.what()), sizeof(message));
						std::memcpy(message, e.what(), message_len);
					} catch(...){
						LOG_POSEIDON_WARNING("Unknown exception thrown");
						except = boost::current_exception();

						err_code = MONGOC_ERROR_PROTOCOL_ERROR;
						message_len = 17;
						std::memcpy(message, "Unknown exception", 17);
					}
					conn->discard_result();
				}

				if(except){
					const AUTO(retry_count, ++elem->retry_count);
					if(retry_count < g_max_retry_count){
						LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO,
							"Going to retry MongoDB operation: retry_count = ", retry_count);
						elem->due_time = now + (g_retry_init_delay << retry_count);
						boost::rethrow_exception(except);
					}

					LOG_POSEIDON_ERROR("Max retry count exceeded.");
					dump_bson_to_file(query, err_code, message, message_len);
					elem->operation->set_exception(except);
				} else {
					elem->operation->set_success();
				}

				const Mutex::UniqueLock lock(m_mutex);
				m_queue.pop_front();
			}
		}

		void dump_bson_to_file(const MongoDb::BsonBuilder &query, long err_code, const char *message, std::size_t message_len){
			PROFILE_ME;

			if(g_dump_dir.empty()){
				LOG_POSEIDON_WARNING("MongoDB dump is disabled.");
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
				const int err_code = errno;
				LOG_POSEIDON_FATAL("Error creating SQL dump file: dump_path = ", dump_path,
					", errno = ", err_code, ", desc = ", get_error_desc(err_code));
				std::abort();
			}

			LOG_POSEIDON_INFO("Writing MongoDB dump...");
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
			dump.append(query.build_json(false));
			dump.append("\n\n");

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
			const Mutex::UniqueLock lock(m_mutex);
			Thread(boost::bind(&MongoDbThread::thread_proc, this), " D  ").swap(m_thread);
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
				MongoDb::BsonBuilder current_bson;
				bool alive;
				{
					const Mutex::UniqueLock lock(m_mutex);
					pending_objects = m_queue.size();
					if(pending_objects == 0){
						break;
					}
					current_bson = m_queue.front().operation->generate_query();
					alive = atomic_load(m_alive, ATOMIC_CONSUME);
					atomic_store(m_urgent, true, ATOMIC_RELEASE);
					m_new_operation.signal();
				}
				LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO,
					"Waiting for SQL queries to complete: pending_objects = ", pending_objects, ", current_bson = ", current_bson);

				if(!alive){
					LOG_POSEIDON_ERROR("MongoDB thread seems dead before the queue is emptied. Trying to recover...");
					try {
						if(m_thread.joinable()){
							m_thread.join();
						}
						LOG_POSEIDON_WARNING("Recreating MongoDB thread: pending_objects = ", pending_objects);
						start();
					} catch(MongoDb::Exception &e){
						LOG_POSEIDON_WARNING("MongoDb::Exception thrown: code = ", e.get_code(), ", what = ", e.what());
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

		void add_operation(boost::shared_ptr<OperationBase> operation, bool urgent){
			PROFILE_ME;

			const AUTO(combinable_object, operation->get_combinable_object());

			AUTO(due_time, get_fast_mono_clock());
			// 有紧急操作时无视写入延迟，这个逻辑不在这里处理。
			due_time += g_save_delay;

			const Mutex::UniqueLock lock(m_mutex);
			if(!atomic_load(m_running, ATOMIC_CONSUME)){
				LOG_POSEIDON_ERROR("MongoDB thread is being shut down.");
				DEBUG_THROW(Exception, sslit("MongoDB thread is being shut down"));
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

	struct CollectionNameComparator {
		bool operator()(const char *lhs, const char *rhs) const {
			return std::strcmp(lhs, rhs) < 0;
		}
	};

	volatile bool g_running = false;

	Mutex g_thread_mutex;
	boost::container::flat_map<const char *, boost::shared_ptr<MongoDbThread>, CollectionNameComparator> g_threads;

	void submit_operation_by_collection(const char *collection, boost::shared_ptr<OperationBase> operation, bool urgent){
		PROFILE_ME;

		boost::shared_ptr<MongoDbThread> thread;
		{
			const Mutex::UniqueLock lock(g_thread_mutex);
			AUTO(it, g_threads.find(collection));
			if(it == g_threads.end()){
				LOG_POSEIDON_INFO("Creating new MongoDB thread: collection = ", collection);
				thread = boost::make_shared<MongoDbThread>();
				thread->start();
				it = g_threads.emplace(collection, thread).first;
			} else {
				thread = it->second;
			}
		}
		thread->add_operation(STD_MOVE(operation), urgent);
	}
	std::size_t submit_operation_all(const boost::shared_ptr<volatile std::size_t> &counter,
		boost::shared_ptr<OperationBase> operation, bool urgent)
	{
		PROFILE_ME;

		VALUE_TYPE(g_threads) threads;
		{
			const Mutex::UniqueLock lock(g_thread_mutex);
			threads = g_threads;
		}
		atomic_store(*counter, threads.size(), ATOMIC_RELAXED);
		for(AUTO(it, threads.begin()); it != threads.end(); ++it){
			it->second->add_operation(operation, urgent);
		}
		return threads.size();
	}
}

void MongoDbDaemon::start(){
	if(atomic_exchange(g_running, true, ATOMIC_ACQ_REL) != false){
		LOG_POSEIDON_FATAL("Only one daemon is allowed at the same time.");
		std::abort();
	}
	LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "Starting MongoDB daemon...");

	MainConfig::get(g_server_addr, "mongodb_server_addr");
	LOG_POSEIDON_DEBUG("MongoDB server addr = ", g_server_addr);

	MainConfig::get(g_server_port, "mongodb_server_port");
	LOG_POSEIDON_DEBUG("MongoDB server port = ", g_server_port);

	MainConfig::get(g_slave_addr, "mongodb_slave_addr");
	LOG_POSEIDON_DEBUG("MongoDB slave addr = ", g_slave_addr);

	MainConfig::get(g_slave_port, "mongodb_slave_port");
	LOG_POSEIDON_DEBUG("MongoDB slave port = ", g_slave_port);

	MainConfig::get(g_username, "mongodb_username");
	LOG_POSEIDON_DEBUG("MongoDB username = ", g_username);

	MainConfig::get(g_password, "mongodb_password");
	LOG_POSEIDON_DEBUG("MongoDB password = ", g_password);

	MainConfig::get(g_auth_database, "mongodb_auth_database");
	LOG_POSEIDON_DEBUG("MongoDB auth database = ", g_auth_database);

	MainConfig::get(g_database, "mongodb_database");
	LOG_POSEIDON_DEBUG("MongoDB database = ", g_database);

	MainConfig::get(g_use_ssl, "mongodb_use_ssl");
	LOG_POSEIDON_DEBUG("MongoDB use ssl = ", g_use_ssl);

	MainConfig::get(g_dump_dir, "mongodb_dump_dir");
	LOG_POSEIDON_DEBUG("MongoDB dump dir = ", g_dump_dir);

	MainConfig::get(g_save_delay, "mongodb_save_delay");
	LOG_POSEIDON_DEBUG("MongoDB save delay = ", g_save_delay);

	MainConfig::get(g_reconn_delay, "mongodb_reconn_delay");
	LOG_POSEIDON_DEBUG("MongoDB reconnect delay = ", g_reconn_delay);

	MainConfig::get(g_max_retry_count, "mongodb_max_retry_count");
	LOG_POSEIDON_DEBUG("MongoDB max retry count = ", g_max_retry_count);

	MainConfig::get(g_retry_init_delay, "mongodb_retry_init_delay");
	LOG_POSEIDON_DEBUG("MongoDB retry init delay = ", g_retry_init_delay);

	if(!g_dump_dir.empty()){
		const AUTO(placeholder_path, g_dump_dir + "/placeholder");
		LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO,
			"Checking whether MongoDB dump directory is writeable: test_path = ", placeholder_path);
		UniqueFile dump_file;
		if(!dump_file.reset(::open(placeholder_path.c_str(), O_WRONLY | O_TRUNC | O_CREAT, 0644))){
			const int err_code = errno;
			LOG_POSEIDON_FATAL("Could not create placeholder file: placeholder_path = ", placeholder_path, ", errno = ", err_code);
			std::abort();
		}
	}

	LOG_POSEIDON_INFO("MongoDB daemon started.");
}
void MongoDbDaemon::stop(){
	if(atomic_exchange(g_running, false, ATOMIC_ACQ_REL) == false){
		return;
	}
	LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "Stopping MongoDB daemon...");

	for(;;){
		VALUE_TYPE(g_threads) threads;
		{
			const Mutex::UniqueLock lock(g_thread_mutex);
			threads.swap(g_threads);
		}
		if(threads.empty()){
			break;
		}
		for(AUTO(it, threads.begin()); it != threads.end(); ++it){
			LOG_POSEIDON_INFO("Stopping MongoDB thread: collection = ", it->first);
			it->second->stop();
		}
		for(AUTO(it, threads.begin()); it != threads.end(); ++it){
			LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO,
				"Waiting for MongoDB thread to terminate: collection = ", it->first);
			it->second->safe_join();
		}
	}

	LOG_POSEIDON_INFO("MongoDB daemon stopped.");
}

boost::shared_ptr<MongoDb::Connection> MongoDbDaemon::create_connection(bool from_slave){
	AUTO(addr, &g_server_addr);
	AUTO(port, &g_server_port);
	if(from_slave){
		if(!g_slave_addr.empty()){
			addr = &g_slave_addr;
		}
		if(g_slave_port != 0){
			port = &g_slave_port;
		}
	}
	return MongoDb::Connection::create(*addr, *port, g_username, g_password, g_auth_database, g_use_ssl, g_database);
}

void MongoDbDaemon::wait_for_all_async_operations(){
	VALUE_TYPE(g_threads) threads;
	{
		const Mutex::UniqueLock lock(g_thread_mutex);
		threads = g_threads;
	}

	for(AUTO(it, threads.begin()); it != threads.end(); ++it){
		it->second->wait_till_idle();
	}
}

boost::shared_ptr<const JobPromise> MongoDbDaemon::enqueue_for_saving(
	boost::shared_ptr<const MongoDb::ObjectBase> object, bool to_replace, bool urgent)
{
	AUTO(promise, boost::make_shared<JobPromise>());
	const char *const collection_name = object->get_collection_name();
	AUTO(operation, boost::make_shared<SaveOperation>(promise, STD_MOVE(object), to_replace));
	submit_operation_by_collection(collection_name, STD_MOVE_IDN(operation), urgent);
	return STD_MOVE_IDN(promise);
}
boost::shared_ptr<const JobPromise> MongoDbDaemon::enqueue_for_loading(
	boost::shared_ptr<MongoDb::ObjectBase> object, MongoDb::BsonBuilder query)
{
	AUTO(promise, boost::make_shared<JobPromise>());
	const char *const collection_name = object->get_collection_name();
	AUTO(operation, boost::make_shared<LoadOperation>(promise, STD_MOVE(object), STD_MOVE(query)));
	submit_operation_by_collection(collection_name, STD_MOVE_IDN(operation), true);
	return STD_MOVE_IDN(promise);
}
boost::shared_ptr<const JobPromise> MongoDbDaemon::enqueue_for_deleting(
	const char *collection, MongoDb::BsonBuilder query, bool delete_all)
{
	AUTO(promise, boost::make_shared<JobPromise>());
	const char *const collection_name = collection;
	AUTO(operation, boost::make_shared<DeleteOperation>(promise, collection, STD_MOVE(query), delete_all));
	submit_operation_by_collection(collection_name, STD_MOVE_IDN(operation), true);
	return STD_MOVE_IDN(promise);
}
boost::shared_ptr<const JobPromise> MongoDbDaemon::enqueue_for_batch_loading(
	ObjectFactory factory, const char *collection, MongoDb::BsonBuilder query, boost::uint32_t begin, boost::uint32_t limit)
{
	AUTO(promise, boost::make_shared<JobPromise>());
	const char *const collection_name = collection;
	AUTO(operation, boost::make_shared<BatchLoadOperation>(promise, STD_MOVE(factory), collection, STD_MOVE(query), begin, limit));
	submit_operation_by_collection(collection_name, STD_MOVE_IDN(operation), true);
	return STD_MOVE_IDN(promise);
}

boost::shared_ptr<const JobPromise> MongoDbDaemon::enqueue_for_waiting_for_all_async_operations(){
	AUTO(promise, boost::make_shared<JobPromise>());
	AUTO(counter, boost::make_shared<volatile std::size_t>());
	AUTO(operation, boost::make_shared<WaitOperation>(promise, counter));
	if(submit_operation_all(counter, STD_MOVE_IDN(operation), true) == 0){
		promise->set_success();
	}
	return STD_MOVE_IDN(promise);
}

}
