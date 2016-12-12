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
#include "../buffer_streams.hpp"

namespace Poseidon {

namespace {
	std::string     g_master_addr       = "localhost";
	unsigned        g_master_port       = 27017;
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
		std::string dump_debug(const MongoDb::BsonBuilder &query) const {
			Buffer_ostream os;
			dump(os, query);
			return os.get_buffer().dump_string();
		}

		virtual bool should_use_slave() const = 0;
		virtual boost::shared_ptr<const MongoDb::ObjectBase> get_combinable_object() const = 0;
		virtual const char *get_collection_name() const = 0;
		virtual void generate_bson(MongoDb::BsonBuilder &query) const = 0;
		virtual void execute(const boost::shared_ptr<MongoDb::Connection> &conn, const MongoDb::BsonBuilder &query) const = 0;
		virtual void dump(std::ostream &os, const MongoDb::BsonBuilder &query) const = 0;
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
		void generate_bson(MongoDb::BsonBuilder &query) const OVERRIDE {
			m_object->generate_document(query);
		}
		void execute(const boost::shared_ptr<MongoDb::Connection> &conn, const MongoDb::BsonBuilder &query) const OVERRIDE {
			PROFILE_ME;

			AUTO(pkey, m_object->generate_primary_key());
			if(pkey.empty()){
				LOG_POSEIDON_DEBUG("Inserting: query = ", query);
				conn->execute_insert(get_collection_name(), query, false);
			} else {
				LOG_POSEIDON_DEBUG("Upserting: pkey = ", pkey, ", query = ", query);
				AUTO(q, MongoDb::bson_scalar_string(sslit("_id"), STD_MOVE(pkey)));
				AUTO(d, MongoDb::bson_scalar_object(sslit("$set"), query));
				conn->execute_update(get_collection_name(), STD_MOVE(q), STD_MOVE(d), true, false);
			}
		}
		void dump(std::ostream &os, const MongoDb::BsonBuilder &query) const OVERRIDE {
			PROFILE_ME;

			AUTO(pkey, m_object->generate_primary_key());
			if(pkey.empty()){
				os <<"db." <<get_collection_name() <<".insert(" <<query <<")";
			} else {
				AUTO(q, MongoDb::bson_scalar_string(sslit("_id"), STD_MOVE(pkey)));
				AUTO(d, MongoDb::bson_scalar_object(sslit("$set"), query));
				os <<"db." <<get_collection_name() <<".update(" <<q <<"," <<d <<",{upsert:1})";
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
		void generate_bson(MongoDb::BsonBuilder &query) const OVERRIDE {
			query = m_query;
		}
		void execute(const boost::shared_ptr<MongoDb::Connection> &conn, const MongoDb::BsonBuilder &query) const OVERRIDE {
			PROFILE_ME;

			if(m_promise.unique()){
				LOG_POSEIDON_DEBUG("Discarding isolated MongoDB query: collection = ", get_collection_name(), ", query = ", query);
				return;
			}

			conn->execute_query(get_collection_name(), query, 0, 1);
			if(!conn->fetch_next()){
				DEBUG_THROW(MongoDb::Exception, SharedNts::view(get_collection_name()), MONGOC_ERROR_QUERY_FAILURE, sslit("No rows returned"));
			}
			m_object->fetch(conn);
		}
		void dump(std::ostream &os, const MongoDb::BsonBuilder &query) const OVERRIDE {
			PROFILE_ME;

			os <<"db." <<get_collection_name() <<".find(" <<query <<").limit(1)";
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
		void generate_bson(MongoDb::BsonBuilder &query) const OVERRIDE {
			query = m_query;
		}
		void execute(const boost::shared_ptr<MongoDb::Connection> &conn, const MongoDb::BsonBuilder &query) const OVERRIDE {
			PROFILE_ME;

			conn->execute_delete(get_collection_name(), query, m_delete_all);
		}
		void dump(std::ostream &os, const MongoDb::BsonBuilder &query) const OVERRIDE {
			PROFILE_ME;

			os <<"db." <<get_collection_name() <<".remove(" <<query <<",{justOne:" <<!m_delete_all <<"})";
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
		const MongoDbDaemon::ObjectFactory m_factory;
		const char *const m_collection;
		const MongoDb::BsonBuilder m_query;
		const boost::uint32_t m_begin;
		const boost::uint32_t m_limit;

	public:
		BatchLoadOperation(boost::shared_ptr<JobPromise> promise,
			MongoDbDaemon::ObjectFactory factory, const char *collection,
			MongoDb::BsonBuilder query, boost::uint32_t begin, boost::uint32_t limit)
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
		void generate_bson(MongoDb::BsonBuilder &query) const OVERRIDE {
			query = m_query;
		}
		void execute(const boost::shared_ptr<MongoDb::Connection> &conn, const MongoDb::BsonBuilder &query) const OVERRIDE {
			PROFILE_ME;

			if(m_promise.unique()){
				LOG_POSEIDON_DEBUG("Discarding isolated MongoDB query: collection = ", get_collection_name(), ", query = ", query);
				return;
			}

			conn->execute_query(get_collection_name(), query, m_begin, m_limit);
			if(m_factory){
				while(conn->fetch_next()){
					m_factory(conn);
				}
			} else {
				LOG_POSEIDON_DEBUG("Result discarded.");
			}
		}
		void dump(std::ostream &os, const MongoDb::BsonBuilder &query) const OVERRIDE {
			PROFILE_ME;

			os <<"db." <<get_collection_name() <<".find(" <<query <<").skip(" <<m_begin <<").limit(" <<m_limit <<")";
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
		boost::shared_ptr<const MongoDb::ObjectBase> get_combinable_object() const OVERRIDE {
			return VAL_INIT; // 不能合并。
		}
		const char *get_collection_name() const OVERRIDE {
			return "";
		}
		void generate_bson(MongoDb::BsonBuilder &query) const OVERRIDE {
			query = MongoDb::bson_scalar_signed(sslit("ping"), 1);
		}
		void execute(const boost::shared_ptr<MongoDb::Connection> &conn, const MongoDb::BsonBuilder &query) const OVERRIDE {
			PROFILE_ME;

			conn->execute_command(get_collection_name(), query, 0, 1);
		}
		void dump(std::ostream &os, const MongoDb::BsonBuilder &query) const OVERRIDE {
			PROFILE_ME;

			os <<"db.runCommand(" <<query <<")";
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
		boost::container::deque<OperationQueueElement> m_queue;

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
						master_conn = MongoDb::Connection::create(g_master_addr, g_master_port,
							g_username, g_password, g_auth_database, g_use_ssl, g_database);
						LOG_POSEIDON_INFO("Successfully connected to MongoDB master server.");
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
						LOG_POSEIDON_INFO("Connecting to MongoDB slave server...");
						try {
							slave_conn = MongoDbDaemon::create_connection(true);
							LOG_POSEIDON_INFO("Successfully connected to MongoDB slave server.");
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

				MongoDb::BsonBuilder query;
				boost::uint32_t err_code = 0;
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
					operation->generate_bson(query);
					try {
						LOG_POSEIDON_DEBUG("Executing MongoDB query: ", operation->dump_debug(query));
						operation->execute(conn, query);
					} catch(MongoDb::Exception &e){
						LOG_POSEIDON_WARNING("MongoDb::Exception thrown: code = ", e.get_code(), ", what = ", e.what());
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

						err_code = MONGOC_ERROR_PROTOCOL_ERROR;
						err_msg = e.what();
					} catch(...){
						LOG_POSEIDON_WARNING("Unknown exception thrown");
#ifdef POSEIDON_CXX11
						except = std::current_exception();
#else
						except = boost::copy_exception(std::bad_exception());
#endif

						err_code = MONGOC_ERROR_PROTOCOL_ERROR;
						err_msg = "Unknown exception";
					}
					conn->discard_result();
				}

				if(except){
					const AUTO(retry_count, ++elem->retry_count);
					if(retry_count < g_max_retry_count){
						LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO,
							"Going to retry MongoDB operation: retry_count = ", retry_count);
						elem->due_time = now + (g_retry_init_delay << retry_count);
#ifdef POSEIDON_CXX11
						std::rethrow_exception(except);
#else
						boost::rethrow_exception(except);
#endif
					}

					LOG_POSEIDON_ERROR("Max retry count exceeded.");
					dump_bson_to_file(elem->operation, query, err_code, err_msg);
					elem->operation->set_exception(except);
				} else {
					elem->operation->set_success();
				}

				const Mutex::UniqueLock lock(m_mutex);
				m_queue.pop_front();
			}
		}

		void dump_bson_to_file(const boost::shared_ptr<OperationBase> &operation,
			const MongoDb::BsonBuilder &query, long err_code, const std::string &err_msg)
		{
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

			LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "Creating BSON dump file: ", dump_path);
			UniqueFile dump_file;
			if(!dump_file.reset(::open(dump_path.c_str(), O_WRONLY | O_APPEND | O_CREAT, 0644))){
				const int saved_errno = errno;
				LOG_POSEIDON_FATAL("Error creating BSON dump file: dump_path = ", dump_path,
					", errno = ", saved_errno, ", desc = ", get_error_desc(saved_errno));
				std::abort();
			}

			LOG_POSEIDON_INFO("Writing MongoDB dump...");
			Buffer_ostream os;
			len = format_time(temp, sizeof(temp), local_now, false);
			os <<"// " <<temp <<": err_code = " <<err_code <<", err_msg = " <<err_msg <<std::endl;
			operation->dump(os, query);
			os <<";" <<std::endl;
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
					m_queue.front().operation->generate_bson(current_bson);
					alive = atomic_load(m_alive, ATOMIC_CONSUME);
					atomic_store(m_urgent, true, ATOMIC_RELEASE);
					m_new_operation.signal();
				}
				LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO,
					"Waiting for BSON queries to complete: pending_objects = ", pending_objects, ", current_bson = ", current_bson);

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

	volatile bool g_running = false;

	Mutex g_router_mutex;
	struct Route {
		boost::shared_ptr<const void> probe;
		boost::shared_ptr<MongoDbThread> thread;
	};
	boost::container::flat_map<SharedNts, Route> g_router;
	boost::container::flat_multimap<std::size_t, std::size_t> g_routing_map;
	std::vector<boost::shared_ptr<MongoDbThread> > g_threads;

	void submit_operation_by_collection(const char *collection, boost::shared_ptr<OperationBase> operation, bool urgent){
		PROFILE_ME;

		boost::shared_ptr<const void> probe;
		boost::shared_ptr<MongoDbThread> thread;
		{
			const Mutex::UniqueLock lock(g_router_mutex);

			AUTO_REF(route, g_router[SharedNts::view(collection)]);
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
						"Creating new MongoDB thread ", i, " for collection ", collection);
					thread = boost::make_shared<MongoDbThread>();
					thread->start();
					test_thread = thread;
					route.thread = thread;
					goto _use_thread;
				}
				const AUTO(queue_size, test_thread->get_queue_size());
				LOG_POSEIDON_DEBUG("> MongoDB thread ", i, "'s queue size: ", queue_size);
				g_routing_map.emplace(queue_size, i);
			}
			if(g_routing_map.empty()){
				LOG_POSEIDON_FATAL("No available MongoDB thread?!");
				std::abort();
			}
			const AUTO(index, g_routing_map.begin()->second);
			LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_DEBUG,
				"Picking thread ", index, " for collection ", collection);
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

void MongoDbDaemon::start(){
	if(atomic_exchange(g_running, true, ATOMIC_ACQ_REL) != false){
		LOG_POSEIDON_FATAL("Only one daemon is allowed at the same time.");
		std::abort();
	}
	LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "Starting MongoDB daemon...");

	MainConfig::get(g_master_addr, "mongodb_server_addr");
	LOG_POSEIDON_DEBUG("MongoDB master addr = ", g_master_addr);

	MainConfig::get(g_master_port, "mongodb_server_port");
	LOG_POSEIDON_DEBUG("MongoDB master port = ", g_master_port);

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

	MainConfig::get(g_max_thread_count, "mongodb_max_thread_count");
	LOG_POSEIDON_DEBUG("MongoDB max_thread_count = ", g_max_thread_count);

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

	g_threads.resize(std::max<std::size_t>(g_max_thread_count, 1));

	LOG_POSEIDON_INFO("MongoDB daemon started.");
}
void MongoDbDaemon::stop(){
	if(atomic_exchange(g_running, false, ATOMIC_ACQ_REL) == false){
		return;
	}
	LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "Stopping MongoDB daemon...");

	for(std::size_t i = 0; i < g_threads.size(); ++i){
		const AUTO_REF(thread, g_threads.at(i));
		if(!thread){
			continue;
		}
		LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "Stopping MongoDB thread ", i);
		thread->stop();
	}
	for(std::size_t i = 0; i < g_threads.size(); ++i){
		const AUTO_REF(thread, g_threads.at(i));
		if(!thread){
			continue;
		}
		LOG_POSEIDON(Logger::SP_MAJOR | Logger::LV_INFO, "Waiting for MongoDB thread ", i, " to terminate...");
		thread->safe_join();
	}
	g_threads.clear();

	LOG_POSEIDON_INFO("MongoDB daemon stopped.");
}

boost::shared_ptr<MongoDb::Connection> MongoDbDaemon::create_connection(bool from_slave){
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
	return MongoDb::Connection::create(*addr, *port, g_username, g_password, g_auth_database, g_use_ssl, g_database);
}

void MongoDbDaemon::wait_for_all_async_operations(){
	for(std::size_t i = 0; i < g_threads.size(); ++i){
		const AUTO_REF(thread, g_threads.at(i));
		if(!thread){
			continue;
		}
		thread->wait_till_idle();
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
	MongoDbDaemon::ObjectFactory factory, const char *collection, MongoDb::BsonBuilder query, boost::uint32_t begin, boost::uint32_t limit)
{
	AUTO(promise, boost::make_shared<JobPromise>());
	const char *const collection_name = collection;
	AUTO(operation, boost::make_shared<BatchLoadOperation>(promise, STD_MOVE(factory), collection, STD_MOVE(query), begin, limit));
	submit_operation_by_collection(collection_name, STD_MOVE_IDN(operation), true);
	return STD_MOVE_IDN(promise);
}

boost::shared_ptr<const JobPromise> MongoDbDaemon::enqueue_for_waiting_for_all_async_operations(){
	AUTO(promise, boost::make_shared<JobPromise>());
	AUTO(operation, boost::make_shared<WaitOperation>(promise));
	submit_operation_all(STD_MOVE_IDN(operation), true);
	return STD_MOVE_IDN(promise);
}

}
