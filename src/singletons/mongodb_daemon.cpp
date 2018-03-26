// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "mongodb_daemon.hpp"
#include "main_config.hpp"
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
#include <libmongoc-1.0/mongoc.h>

namespace Poseidon {

typedef Mongodb_daemon::Query_callback Query_callback;

namespace {
	boost::shared_ptr<Mongodb::Connection> real_create_connection(bool from_slave, const boost::shared_ptr<Mongodb::Connection> &master_conn){
		std::string server_addr;
		boost::uint16_t server_port = 0;
		if(from_slave){
			server_addr = Main_config::get<std::string>("mongodb_slave_addr");
			server_port = Main_config::get<boost::uint16_t>("mongodb_slave_port");
		}
		if(server_addr.empty()){
			if(master_conn){
				LOG_POSEIDON_DEBUG("MongoDB slave is not configured. Reuse the master connection as a slave.");
				return master_conn;
			}
			server_addr = Main_config::get<std::string>("mongodb_server_addr", "localhost");
			server_port = Main_config::get<boost::uint16_t>("mongodb_server_port", 27017);
		}
		std::string username = Main_config::get<std::string>("mongodb_username", "root");
		std::string password = Main_config::get<std::string>("mongodb_password");
		std::string auth_db = Main_config::get<std::string>("mongodb_auth_database", "admin");
		bool use_ssl = Main_config::get<bool>("mongodb_use_ssl", false);
		std::string database = Main_config::get<std::string>("mongodb_database", "poseidon");
		return Mongodb::Connection::create(server_addr.c_str(), server_port, username.c_str(), password.c_str(), auth_db.c_str(), use_ssl, database.c_str());
	}

	// 对于日志文件的写操作应当互斥。
	Mutex g_dump_mutex;

	void dump_bson_to_file(const Mongodb::Bson_builder &query, unsigned long err_code, const char *err_msg) NOEXCEPT
	try {
		PROFILE_ME;

		const AUTO(dump_dir, Main_config::get<std::string>("mongodb_dump_dir"));
		if(dump_dir.empty()){
			LOG_POSEIDON_WARNING("MongoDB dump is disabled.");
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

		LOG_POSEIDON(Logger::special_major | Logger::level_info, "Creating BSON dump file: ", dump_path);
		Unique_file dump_file;
		if(!dump_file.reset(::open(dump_path.c_str(), O_WRONLY | O_APPEND | O_CREAT, 0644))){
			const int saved_errno = errno;
			LOG_POSEIDON_FATAL("Error creating BSON dump file: dump_path = ", dump_path, ", errno = ", saved_errno, ", desc = ", get_error_desc(saved_errno));
			std::abort();
		}

		LOG_POSEIDON(Logger::special_major | Logger::level_info, "Writing MongoDB dump...");
		Buffer_ostream os;
		len = format_time(temp, sizeof(temp), local_now, false);
		os <<"// " <<temp <<": err_code = " <<err_code <<", err_msg = " <<err_msg <<std::endl;
		if(query.empty()){
			os <<"// <low level access>";
		} else {
			os <<"db.runCommand(" <<query <<");";
		}
		os <<std::endl <<std::endl;
		const AUTO(str, os.get_buffer().dump_string());

		const Mutex::Unique_lock lock(g_dump_mutex);
		std::size_t total = 0;
		do {
			::ssize_t written = ::write(dump_file.get(), str.data() + total, str.size() - total);
			if(written <= 0){
				break;
			}
			total += static_cast<std::size_t>(written);
		} while(total < str.size());
	} catch(std::exception &e){
		LOG_POSEIDON_ERROR("Error writing BSON dump: what = ", e.what());
	}

	// 数据库线程操作。
	class Operation_base : NONCOPYABLE {
	private:
		const boost::weak_ptr<Promise> m_weak_promise;

		boost::shared_ptr<const void> m_probe;

	public:
		explicit Operation_base(const boost::shared_ptr<Promise> &promise)
			: m_weak_promise(promise)
		{
			//
		}
		virtual ~Operation_base(){
			//
		}

	public:
		void set_probe(boost::shared_ptr<const void> probe){
			m_probe = STD_MOVE(probe);
		}

		virtual boost::shared_ptr<Promise> get_promise() const {
			return m_weak_promise.lock();
		}
		virtual bool should_use_slave() const = 0;
		virtual boost::shared_ptr<const Mongodb::Object_base> get_combinable_object() const = 0;
		virtual const char *get_collection() const = 0;
		virtual void generate_bson(Mongodb::Bson_builder &query) const = 0;
		virtual void execute(const boost::shared_ptr<Mongodb::Connection> &conn, const Mongodb::Bson_builder &query) = 0;
	};

	class Save_operation : public Operation_base {
	private:
		boost::shared_ptr<const Mongodb::Object_base> m_object;
		bool m_to_replace;

	public:
		Save_operation(const boost::shared_ptr<Promise> &promise, boost::shared_ptr<const Mongodb::Object_base> object, bool to_replace)
			: Operation_base(promise)
			, m_object(STD_MOVE(object)), m_to_replace(to_replace)
		{
			//
		}

	protected:
		bool should_use_slave() const OVERRIDE {
			return false;
		}
		boost::shared_ptr<const Mongodb::Object_base> get_combinable_object() const OVERRIDE {
			return m_object;
		}
		const char *get_collection() const OVERRIDE {
			return m_object->get_collection();
		}
		void generate_bson(Mongodb::Bson_builder &query) const OVERRIDE {
			Mongodb::Bson_builder q;
			{
				Mongodb::Bson_builder doc;
				m_object->generate_document(doc);
				AUTO(pkey, m_object->generate_primary_key());
				if(m_to_replace && !pkey.empty()){
					Mongodb::Bson_builder upd;
					upd.append_object(sslit("q"), Mongodb::bson_scalar_string(sslit("_id"), STD_MOVE(pkey)));
					upd.append_object(sslit("u"), STD_MOVE(doc));
					upd.append_boolean(sslit("upsert"), true);
					LOG_POSEIDON_DEBUG("Upserting: pkey = ", pkey, ", upd = ", upd);
					q.append_string(sslit("update"), get_collection());
					q.append_array(sslit("updates"), Mongodb::bson_scalar_object(sslit("0"), STD_MOVE(upd)));
				} else {
					LOG_POSEIDON_DEBUG("Inserting: pkey = ", pkey, ", doc = ", doc);
					q.append_string(sslit("insert"), get_collection());
					q.append_array(sslit("documents"), Mongodb::bson_scalar_object(sslit("0"), STD_MOVE(doc)));
				}
			}
			query = q;
		}
		void execute(const boost::shared_ptr<Mongodb::Connection> &conn, const Mongodb::Bson_builder &query) OVERRIDE {
			PROFILE_ME;

			conn->execute_bson(query);
		}
	};

	class Load_operation : public Operation_base {
	private:
		boost::shared_ptr<Mongodb::Object_base> m_object;
		Mongodb::Bson_builder m_query;

	public:
		Load_operation(const boost::shared_ptr<Promise> &promise, boost::shared_ptr<Mongodb::Object_base> object, Mongodb::Bson_builder query)
			: Operation_base(promise)
			, m_object(STD_MOVE(object)), m_query(STD_MOVE(query))
		{
			//
		}

	protected:
		bool should_use_slave() const OVERRIDE {
			return true;
		}
		boost::shared_ptr<const Mongodb::Object_base> get_combinable_object() const OVERRIDE {
			return VAL_INIT; // 不能合并。
		}
		const char *get_collection() const OVERRIDE {
			return m_object->get_collection();
		}
		void generate_bson(Mongodb::Bson_builder &query) const OVERRIDE {
			query = m_query;
		}
		void execute(const boost::shared_ptr<Mongodb::Connection> &conn, const Mongodb::Bson_builder &query) OVERRIDE {
			PROFILE_ME;

			if(!get_promise()){
				LOG_POSEIDON_WARNING("Discarding isolated MongoDB query: collection = ", get_collection(), ", query = ", query);
				return;
			}
			conn->execute_bson(query);
			DEBUG_THROW_UNLESS(conn->fetch_document(), Mongodb::Exception, Shared_nts::view(get_collection()), MONGOC_ERROR_QUERY_FAILURE, sslit("No documents returned"));
			m_object->fetch(conn);
		}
	};

	class Delete_operation : public Operation_base {
	private:
		const char *m_collection;
		Mongodb::Bson_builder m_query;

	public:
		Delete_operation(const boost::shared_ptr<Promise> &promise, const char *collection, Mongodb::Bson_builder query)
			: Operation_base(promise)
			, m_collection(collection), m_query(STD_MOVE(query))
		{
			//
		}

	protected:
		bool should_use_slave() const OVERRIDE {
			return false;
		}
		boost::shared_ptr<const Mongodb::Object_base> get_combinable_object() const OVERRIDE {
			return VAL_INIT; // 不能合并。
		}
		const char *get_collection() const OVERRIDE {
			return m_collection;
		}
		void generate_bson(Mongodb::Bson_builder &query) const OVERRIDE {
			query = m_query;
		}
		void execute(const boost::shared_ptr<Mongodb::Connection> &conn, const Mongodb::Bson_builder &query) OVERRIDE {
			PROFILE_ME;

			conn->execute_bson(query);
		}
	};

	class Batch_load_operation : public Operation_base {
	private:
		Query_callback m_callback;
		const char *m_collection_hint;
		Mongodb::Bson_builder m_query;

	public:
		Batch_load_operation(const boost::shared_ptr<Promise> &promise, Query_callback callback, const char *collection_hint, Mongodb::Bson_builder query)
			: Operation_base(promise)
			, m_callback(STD_MOVE_IDN(callback)), m_collection_hint(collection_hint), m_query(STD_MOVE(query))
		{
			//
		}

	protected:
		bool should_use_slave() const OVERRIDE {
			return true;
		}
		boost::shared_ptr<const Mongodb::Object_base> get_combinable_object() const OVERRIDE {
			return VAL_INIT; // 不能合并。
		}
		const char *get_collection() const OVERRIDE {
			return m_collection_hint;
		}
		void generate_bson(Mongodb::Bson_builder &query) const OVERRIDE {
			query = m_query;
		}
		void execute(const boost::shared_ptr<Mongodb::Connection> &conn, const Mongodb::Bson_builder &query) OVERRIDE {
			PROFILE_ME;

			if(!get_promise()){
				LOG_POSEIDON_WARNING("Discarding isolated MongoDB query: collection = ", get_collection(), ", query = ", query);
				return;
			}
			conn->execute_bson(query);
			if(m_callback){
				while(conn->fetch_document()){
					m_callback(conn);
				}
			} else {
				LOG_POSEIDON_DEBUG("Result discarded.");
			}
		}
	};

	class Low_level_access_operation : public Operation_base {
	private:
		Query_callback m_callback;
		const char *m_collection_hint;
		bool m_from_slave;

	public:
		Low_level_access_operation(const boost::shared_ptr<Promise> &promise, Query_callback callback, const char *collection_hint, bool from_slave)
			: Operation_base(promise)
			, m_callback(STD_MOVE_IDN(callback)), m_collection_hint(collection_hint), m_from_slave(from_slave)
		{
			//
		}

	protected:
		bool should_use_slave() const OVERRIDE {
			return m_from_slave;
		}
		boost::shared_ptr<const Mongodb::Object_base> get_combinable_object() const OVERRIDE {
			return VAL_INIT; // 不能合并。
		}
		const char *get_collection() const OVERRIDE {
			return m_collection_hint;
		}
		void generate_bson(Mongodb::Bson_builder & /* query */) const OVERRIDE {
			// no query
		}
		void execute(const boost::shared_ptr<Mongodb::Connection> &conn, const Mongodb::Bson_builder & /* query */) OVERRIDE {
			PROFILE_ME;

			m_callback(conn);
		}
	};

	class Wait_operation : public Operation_base {
	public:
		explicit Wait_operation(const boost::shared_ptr<Promise> &promise)
			: Operation_base(promise)
		{
			//
		}
		~Wait_operation() OVERRIDE {
			const AUTO(promise, Operation_base::get_promise());
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
		boost::shared_ptr<const Mongodb::Object_base> get_combinable_object() const OVERRIDE {
			return VAL_INIT; // 不能合并。
		}
		const char *get_collection() const OVERRIDE {
			return "";
		}
		void generate_bson(Mongodb::Bson_builder &query) const OVERRIDE {
			query = Mongodb::bson_scalar_signed(sslit("ping"), 1);
		}
		void execute(const boost::shared_ptr<Mongodb::Connection> &conn, const Mongodb::Bson_builder &query) OVERRIDE {
			PROFILE_ME;

			conn->execute_bson(query);
		}
	};

	class Mongodb_thread : NONCOPYABLE {
	private:
		struct Operation_queue_element {
			boost::shared_ptr<Operation_base> operation;
			boost::uint64_t due_time;
			std::size_t retry_count;
		};

	private:
		Thread m_thread;
		volatile bool m_running;

		mutable Mutex m_mutex;
		mutable Condition_variable m_new_operation;
		volatile bool m_urgent; // 无视延迟写入，一次性处理队列中所有操作。
		boost::container::deque<Operation_queue_element> m_queue;

	public:
		Mongodb_thread()
			: m_running(false)
			, m_urgent(false)
		{
			//
		}

	private:
		bool pump_one_operation(boost::shared_ptr<Mongodb::Connection> &master_conn, boost::shared_ptr<Mongodb::Connection> &slave_conn) NOEXCEPT {
			PROFILE_ME;

			const AUTO(now, get_fast_mono_clock());
			Operation_queue_element *elem;
			{
				const Mutex::Unique_lock lock(m_mutex);
				if(m_queue.empty()){
					atomic_store(m_urgent, false, memory_order_relaxed);
					return false;
				}
				if(!atomic_load(m_urgent, memory_order_consume) && (now < m_queue.front().due_time)){
					return false;
				}
				elem = &m_queue.front();
			}
			const AUTO_REF(operation, elem->operation);
			AUTO_REF(conn, elem->operation->should_use_slave() ? slave_conn : master_conn);

			Mongodb::Bson_builder query;
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
					operation->generate_bson(query);
					LOG_POSEIDON_DEBUG("Executing MongoDB query: collection = ", operation->get_collection(), ", query = ", query);
					operation->execute(conn, query);
				} catch(Mongodb::Exception &e){
					LOG_POSEIDON_WARNING("Mongodb::Exception thrown: code = ", e.get_code(), ", what = ", e.what());
					except = STD_CURRENT_EXCEPTION();
					err_code = e.get_code();
					::snprintf(err_msg, sizeof(err_msg), "Mongodb::Exception: %s", e.what());
				} catch(std::exception &e){
					LOG_POSEIDON_WARNING("std::exception thrown: what = ", e.what());
					except = STD_CURRENT_EXCEPTION();
					err_code = MONGOC_ERROR_PROTOCOL_ERROR;
					::snprintf(err_msg, sizeof(err_msg), "std::exception: %s", e.what());
				} catch(...){
					LOG_POSEIDON_WARNING("Unknown exception thrown");
					except = STD_CURRENT_EXCEPTION();
					err_code = MONGOC_ERROR_PROTOCOL_ERROR;
					::strcpy(err_msg, "Unknown exception");
				}
				conn->discard_result();
			}
			if(except){
				const AUTO(max_retry_count, Main_config::get<std::size_t>("mongodb_max_retry_count", 3));
				const AUTO(retry_count, ++(elem->retry_count));
				if(retry_count < max_retry_count){
					LOG_POSEIDON(Logger::special_major | Logger::level_info, "Going to retry MongoDB operation: retry_count = ", retry_count);
					const AUTO(retry_init_delay, Main_config::get<boost::uint64_t>("mongodb_retry_init_delay", 1000));
					elem->due_time = now + (retry_init_delay << retry_count);
					conn.reset();
					return true;
				}
				LOG_POSEIDON_ERROR("Max retry count exceeded.");
				dump_bson_to_file(query, err_code, err_msg);
			}
			const AUTO(promise, elem->operation->get_promise());
			if(promise){
				if(except){
					promise->set_exception(STD_MOVE(except), false);
				} else {
					promise->set_success(false);
				}
			}
			const Mutex::Unique_lock lock(m_mutex);
			m_queue.pop_front();
			return true;
		}

		void thread_proc(){
			PROFILE_ME;
			LOG_POSEIDON(Logger::special_major | Logger::level_info, "MongoDB thread started.");

			boost::shared_ptr<Mongodb::Connection> master_conn, slave_conn;
			unsigned timeout = 0;
			for(;;){
				const AUTO(reconnect_delay, Main_config::get<boost::uint64_t>("mongodb_reconn_delay", 5000));
				bool busy;
				do {
					while(!master_conn){
						LOG_POSEIDON(Logger::special_major | Logger::level_info, "Connecting to MongoDB master server...");
						try {
							master_conn = real_create_connection(false, VAL_INIT);
							LOG_POSEIDON(Logger::special_major | Logger::level_info, "Successfully connected to MongoDB master server.");
						} catch(std::exception &e){
							LOG_POSEIDON_ERROR("std::exception thrown: what = ", e.what());
							::timespec req;
							req.tv_sec = (::time_t)(reconnect_delay / 1000);
							req.tv_nsec = (long)(reconnect_delay % 1000) * 1000 * 1000;
							::nanosleep(&req, NULLPTR);
						}
					}
					while(!slave_conn){
						LOG_POSEIDON(Logger::special_major | Logger::level_info, "Connecting to MongoDB slave server...");
						try {
							slave_conn = real_create_connection(true, master_conn);
							LOG_POSEIDON(Logger::special_major | Logger::level_info, "Successfully connected to MongoDB slave server.");
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

				Mutex::Unique_lock lock(m_mutex);
				if(m_queue.empty() && !atomic_load(m_running, memory_order_consume)){
					break;
				}
				m_new_operation.timed_wait(lock, timeout);
			}

			LOG_POSEIDON(Logger::special_major | Logger::level_info, "MongoDB thread stopped.");
		}

	public:
		void start(){
			const Mutex::Unique_lock lock(m_mutex);
			Thread(boost::bind(&Mongodb_thread::thread_proc, this), sslit(" G  "), sslit("MongoDB")).swap(m_thread);
			atomic_store(m_running, true, memory_order_release);
		}
		void stop(){
			atomic_store(m_running, false, memory_order_release);
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
				Mongodb::Bson_builder current_bson;
				{
					const Mutex::Unique_lock lock(m_mutex);
					pending_objects = m_queue.size();
					if(pending_objects == 0){
						break;
					}
					m_queue.front().operation->generate_bson(current_bson);
					atomic_store(m_urgent, true, memory_order_release);
					m_new_operation.signal();
				}
				LOG_POSEIDON(Logger::special_major | Logger::level_info, "Waiting for BSON queries to complete: pending_objects = ", pending_objects, ", current_bson = ", current_bson);

				::timespec req;
				req.tv_sec = 0;
				req.tv_nsec = 500 * 1000 * 1000;
				::nanosleep(&req, NULLPTR);
			}
		}

		std::size_t get_queue_size() const {
			const Mutex::Unique_lock lock(m_mutex);
			return m_queue.size();
		}
		void add_operation(boost::shared_ptr<Operation_base> operation, bool urgent){
			PROFILE_ME;

			const AUTO(combinable_object, operation->get_combinable_object());

			const AUTO(now, get_fast_mono_clock());
			const AUTO(save_delay, Main_config::get<boost::uint64_t>("mongodb_save_delay", 5000));
			// 有紧急操作时无视写入延迟，这个逻辑不在这里处理。
			const AUTO(due_time, saturated_add(now, save_delay));

			const Mutex::Unique_lock lock(m_mutex);
			DEBUG_THROW_UNLESS(atomic_load(m_running, memory_order_consume), Exception, sslit("MongoDB thread is being shut down"));
			Operation_queue_element elem = { STD_MOVE(operation), due_time };
			m_queue.push_back(STD_MOVE(elem));
			if(combinable_object){
				const AUTO(old_write_stamp, combinable_object->get_combined_write_stamp());
				if(!old_write_stamp){
					combinable_object->set_combined_write_stamp(&m_queue.back());
				}
			}
			if(urgent){
				atomic_store(m_urgent, true, memory_order_release);
			}
			m_new_operation.signal();
		}
	};

	volatile bool g_running = false;

	Mutex g_router_mutex;
	struct Route {
		boost::shared_ptr<const void> probe;
		boost::shared_ptr<Mongodb_thread> thread;
	};
	boost::container::flat_map<Shared_nts, Route> g_router;
	boost::container::flat_multimap<std::size_t, std::size_t> g_routing_map;
	boost::container::vector<boost::shared_ptr<Mongodb_thread> > g_threads;

	void add_operation_by_collection(const char *collection, boost::shared_ptr<Operation_base> operation, bool urgent){
		PROFILE_ME;
		DEBUG_THROW_UNLESS(!g_threads.empty(), Basic_exception, sslit("MongoDB support is not enabled"));

		boost::shared_ptr<const void> probe;
		boost::shared_ptr<Mongodb_thread> thread;
		{
			const Mutex::Unique_lock lock(g_router_mutex);

			AUTO_REF(route, g_router[Shared_nts::view(collection)]);
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
					LOG_POSEIDON(Logger::special_major | Logger::level_debug, "Creating new MongoDB thread ", i, " for collection ", collection);
					thread = boost::make_shared<Mongodb_thread>();
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
			LOG_POSEIDON(Logger::special_major | Logger::level_debug, "Picking thread ", index, " for collection ", collection);
			thread = g_threads.at(index);
			route.thread = thread;
		}
	_use_thread:
		assert(probe);
		assert(thread);
		operation->set_probe(STD_MOVE(probe));
		thread->add_operation(STD_MOVE(operation), urgent);
	}
	void add_operation_all(boost::shared_ptr<Operation_base> operation, bool urgent){
		PROFILE_ME;
		DEBUG_THROW_UNLESS(!g_threads.empty(), Basic_exception, sslit("MongoDB support is not enabled"));

		const Mutex::Unique_lock lock(g_router_mutex);
		for(AUTO(it, g_threads.begin()); it != g_threads.end(); ++it){
			const AUTO_REF(thread, *it);
			if(!thread){
				continue;
			}
			thread->add_operation(operation, urgent);
		}
	}
}

void Mongodb_daemon::start(){
	if(atomic_exchange(g_running, true, memory_order_acq_rel) != false){
		LOG_POSEIDON_FATAL("Only one daemon is allowed at the same time.");
		std::abort();
	}
	LOG_POSEIDON(Logger::special_major | Logger::level_info, "Starting MongoDB daemon...");

	const AUTO(max_thread_count, Main_config::get<std::size_t>("mongodb_max_thread_count"));
	if(max_thread_count == 0){
		LOG_POSEIDON_WARNING("MongoDB support has been disabled. To enable MongoDB support, set `mongodb_max_thread_count` in `main.conf` to a value greater than zero.");
	} else {
		boost::shared_ptr<Mongodb::Connection> master_conn, slave_conn;
		LOG_POSEIDON(Logger::special_major | Logger::level_info, "Checking whether MongoDB master server is up...");
		try {
			master_conn = real_create_connection(false, VAL_INIT);
			master_conn->execute_bson(Mongodb::bson_scalar_signed(sslit("ping"), 1));
		} catch(std::exception &e){
			LOG_POSEIDON_FATAL("Could not connect to MongoDB master server: ", e.what());
			LOG_POSEIDON_WARNING("To disable MongoDB support, set `mongodb_max_thread_count` in `main.conf` to zero.");
			std::abort();
		}

		LOG_POSEIDON(Logger::special_major | Logger::level_info, "Checking whether MongoDB slave server is up...");
		try {
			slave_conn = real_create_connection(true, master_conn);
			if(slave_conn != master_conn){
				slave_conn->execute_bson(Mongodb::bson_scalar_signed(sslit("ping"), 1));
			}
		} catch(std::exception &e){
			LOG_POSEIDON_FATAL("Could not connect to MongoDB slave server: ", e.what());
			LOG_POSEIDON_WARNING("To disable MongoDB support, set `mongodb_max_thread_count` in `main.conf` to zero.");
			std::abort();
		}

		const AUTO(dump_dir, Main_config::get<std::string>("mongodb_dump_dir"));
		if(dump_dir.empty()){
			LOG_POSEIDON_WARNING("MongoDB error dump has been disabled. To enable MongoDB error dump, set `mongodb_dump_dir` in `main.conf` to the path to the dump directory.");
		} else {
			LOG_POSEIDON(Logger::special_major | Logger::level_info, "Checking whether MongoDB dump directory is writeable...");
			try {
				const AUTO(placeholder_path, dump_dir + "/placeholder");
				DEBUG_THROW_ASSERT(Unique_file(::open(placeholder_path.c_str(), O_WRONLY | O_TRUNC | O_CREAT, 0644)));
			} catch(std::exception &e){
				LOG_POSEIDON_FATAL("Could not write MongoDB dump: ", e.what());
				LOG_POSEIDON_WARNING("To disable MongoDB error dump, set `mongodb_dump_dir` in `main.conf` to an empty string.");
				std::abort();
			}
		}
	}
	g_threads.resize(max_thread_count);

	LOG_POSEIDON(Logger::special_major | Logger::level_info, "MongoDB daemon started.");
}
void Mongodb_daemon::stop(){
	if(atomic_exchange(g_running, false, memory_order_acq_rel) == false){
		return;
	}
	LOG_POSEIDON(Logger::special_major | Logger::level_info, "Stopping MongoDB daemon...");

	for(std::size_t i = 0; i < g_threads.size(); ++i){
		const AUTO_REF(thread, g_threads.at(i));
		if(!thread){
			continue;
		}
		LOG_POSEIDON(Logger::special_major | Logger::level_info, "Stopping MongoDB thread ", i);
		thread->stop();
	}
	for(std::size_t i = 0; i < g_threads.size(); ++i){
		const AUTO_REF(thread, g_threads.at(i));
		if(!thread){
			continue;
		}
		LOG_POSEIDON(Logger::special_major | Logger::level_info, "Waiting for MongoDB thread ", i, " to terminate...");
		thread->safe_join();
	}

	LOG_POSEIDON(Logger::special_major | Logger::level_info, "MongoDB daemon stopped.");

	const Mutex::Unique_lock lock(g_router_mutex);
	g_threads.clear();
}

boost::shared_ptr<Mongodb::Connection> Mongodb_daemon::create_connection(bool from_slave){
	return real_create_connection(from_slave, VAL_INIT);
}

void Mongodb_daemon::wait_for_all_async_operations(){
	for(std::size_t i = 0; i < g_threads.size(); ++i){
		const AUTO_REF(thread, g_threads.at(i));
		if(!thread){
			continue;
		}
		thread->wait_till_idle();
	}
}

boost::shared_ptr<const Promise> Mongodb_daemon::enqueue_for_saving(boost::shared_ptr<const Mongodb::Object_base> object, bool to_replace, bool urgent){
	AUTO(promise, boost::make_shared<Promise>());
	const char *const collection = object->get_collection();
	AUTO(operation, boost::make_shared<Save_operation>(promise, STD_MOVE(object), to_replace));
	add_operation_by_collection(collection, STD_MOVE_IDN(operation), urgent);
	return STD_MOVE_IDN(promise);
}
boost::shared_ptr<const Promise> Mongodb_daemon::enqueue_for_loading(boost::shared_ptr<Mongodb::Object_base> object, Mongodb::Bson_builder query){
	DEBUG_THROW_ASSERT(!query.empty());

	AUTO(promise, boost::make_shared<Promise>());
	const char *const collection = object->get_collection();
	AUTO(operation, boost::make_shared<Load_operation>(promise, STD_MOVE(object), STD_MOVE(query)));
	add_operation_by_collection(collection, STD_MOVE_IDN(operation), true);
	return STD_MOVE_IDN(promise);
}
boost::shared_ptr<const Promise> Mongodb_daemon::enqueue_for_deleting(const char *collection_hint, Mongodb::Bson_builder query){
	DEBUG_THROW_ASSERT(!query.empty());

	AUTO(promise, boost::make_shared<Promise>());
	const char *const collection = collection_hint;
	AUTO(operation, boost::make_shared<Delete_operation>(promise, collection_hint, STD_MOVE(query)));
	add_operation_by_collection(collection, STD_MOVE_IDN(operation), true);
	return STD_MOVE_IDN(promise);
}
boost::shared_ptr<const Promise> Mongodb_daemon::enqueue_for_batch_loading(Query_callback callback, const char *collection_hint, Mongodb::Bson_builder query){
	DEBUG_THROW_ASSERT(!query.empty());

	AUTO(promise, boost::make_shared<Promise>());
	const char *const collection = collection_hint;
	AUTO(operation, boost::make_shared<Batch_load_operation>(promise, STD_MOVE(callback), collection_hint, STD_MOVE(query)));
	add_operation_by_collection(collection, STD_MOVE_IDN(operation), true);
	return STD_MOVE_IDN(promise);
}

void Mongodb_daemon::enqueue_for_low_level_access(const boost::shared_ptr<Promise> &promise, Query_callback callback, const char *collection_hint, bool from_slave){
	const char *const collection = collection_hint;
	AUTO(operation, boost::make_shared<Low_level_access_operation>(promise, STD_MOVE(callback), collection_hint, from_slave));
	add_operation_by_collection(collection, STD_MOVE_IDN(operation), true);
}

boost::shared_ptr<const Promise> Mongodb_daemon::enqueue_for_waiting_for_all_async_operations(){
	AUTO(promise, boost::make_shared<Promise>());
	AUTO(operation, boost::make_shared<Wait_operation>(promise));
	add_operation_all(STD_MOVE_IDN(operation), true);
	return STD_MOVE_IDN(promise);
}

}
