// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2017, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "connection.hpp"

#if __GNUC__ >= 6
#  pragma GCC diagnostic ignored "-Wignored-attributes"
#  pragma GCC diagnostic push
#  pragma GCC diagnostic ignored "-Wsign-conversion"
#endif

#include "exception.hpp"
#include "bson_builder.hpp"
#include "../raii.hpp"
#include "../log.hpp"
#include "../time.hpp"
#include "../uuid.hpp"
#include <cstdlib>
#include <mongoc.h>
#include <bson.h>

#if __GNUC__ >= 6
#  pragma GCC diagnostic pop
#endif

namespace Poseidon {

namespace MongoDb {
	namespace {
		::pthread_once_t g_mongo_once = PTHREAD_ONCE_INIT;

		void init_mongo(){
			LOG_POSEIDON_INFO("Initializing MongoDB client...");

			::mongoc_init();

			std::atexit(&::mongoc_cleanup);
		}

		struct UriCloser {
			CONSTEXPR ::mongoc_uri_t *operator()() const NOEXCEPT {
				return NULLPTR;
			}
			void operator()(::mongoc_uri_t *uri) const NOEXCEPT {
				::mongoc_uri_destroy(uri);
			}
		};
		struct ClientCloser {
			CONSTEXPR ::mongoc_client_t *operator()() const NOEXCEPT {
				return NULLPTR;
			}
			void operator()(::mongoc_client_t *client) const NOEXCEPT {
				::mongoc_client_destroy(client);
			}
		};

		struct BsonCloser {
			CONSTEXPR ::bson_t *operator()() const NOEXCEPT {
				return NULLPTR;
			}
			void operator()(::bson_t *bson) const NOEXCEPT {
				::bson_destroy(bson);
			}
		};

#define DEBUG_THROW_MONGODB_EXCEPTION(bson_err_, database_)	\
		DEBUG_THROW(::Poseidon::MongoDb::Exception, database_, (bson_err_).code, ::Poseidon::SharedNts((bson_err_).message))

		class DelegatedConnection : public Connection {
		private:
			const SharedNts m_database;

			UniqueHandle<UriCloser> m_uri;
			UniqueHandle<ClientCloser> m_client;

			boost::int64_t m_cursor_id;
			std::string m_cursor_ns;
			UniqueHandle<BsonCloser> m_batch_guard;
			::bson_iter_t m_batch_it;
			::bson_t m_element_storage;
			UniqueHandle<BsonCloser> m_element_guard;

		public:
			DelegatedConnection(const char *server_addr, unsigned server_port,
				const char *user_name, const char *password, const char *auth_database, bool use_ssl, const char *database)
				: m_database(database)
				, m_cursor_id(0), m_cursor_ns()
			{
				const int err = ::pthread_once(&g_mongo_once, &init_mongo);
				if(err != 0){
					LOG_POSEIDON_FATAL("::pthread_once() failed with error code ", err);
					std::abort();
				}

				if(!m_uri.reset(::mongoc_uri_new_for_host_port(server_addr, server_port))){
					DEBUG_THROW(BasicException, sslit("::mongoc_uri_new_for_host_port() failed!"));
				}
				if(!::mongoc_uri_set_username(m_uri.get(), user_name)){
					DEBUG_THROW(BasicException, sslit("::mongoc_uri_set_username() failed!"));
				}
				if(!::mongoc_uri_set_password(m_uri.get(), password)){
					DEBUG_THROW(BasicException, sslit("::mongoc_uri_set_password() failed!"));
				}
				if(!::mongoc_uri_set_database(m_uri.get(), auth_database)){
					DEBUG_THROW(BasicException, sslit("::mongoc_uri_set_database() failed!"));
				}
				if(!::mongoc_uri_set_option_as_bool(m_uri.get(), "ssl", use_ssl)){
					DEBUG_THROW(BasicException, sslit("::mongoc_uri_set_option_as_bool() failed!"));
				}

				if(!m_client.reset(::mongoc_client_new_from_uri(m_uri.get()))){
					DEBUG_THROW(BasicException, sslit("::mongoc_client_new_from_uri() failed!"));
				}
			}

		private:
			bool find_bson_element_and_check_type(::bson_iter_t &it, const char *name, ::bson_type_t type_expecting) const {
				if(!m_element_guard){
					LOG_POSEIDON_WARNING("No more results available.");
					return false;
				}
				const AUTO(element, m_element_guard.get());
				if(!::bson_iter_init_find(&it, element, name)){
					LOG_POSEIDON_WARNING("Field not found: name = ", name);
					return false;
				}
				const AUTO(type, ::bson_iter_type(&it));
				if((type == BSON_TYPE_UNDEFINED) || (type == BSON_TYPE_NULL)){
					LOG_POSEIDON_DEBUG("Field is undefined or null: name = ", name);
					return false;
				}
				if(type != type_expecting){
					LOG_POSEIDON_ERROR("BSON type mismatch: name = ", name, ", type_expecting = ", type_expecting, ", type = ", type);
					DEBUG_THROW(BasicException, sslit("BSON type mismatch"));
				}
				return true;
			}

		public:
			void do_execute_bson(const BsonBuilder &bson){
				const AUTO(query_data, bson.build(false));
				::bson_t query_storage;
				bool success = ::bson_init_static(&query_storage, reinterpret_cast<const boost::uint8_t *>(query_data.data()), query_data.size());
				DEBUG_THROW_ASSERT(success);
				const UniqueHandle<BsonCloser> query_guard(&query_storage);
				const AUTO(query_bt, query_guard.get());

				do_discard_result();

				::bson_t reply_storage;
				::bson_error_t err;
				success = ::mongoc_client_command_simple(m_client.get(), m_database.get(), query_bt, NULLPTR, &reply_storage, &err);
				// `reply` is always set.
				const UniqueHandle<BsonCloser> reply_guard(&reply_storage);
				const AUTO(reply_bt, reply_guard.get());
				if(!success){
					DEBUG_THROW_MONGODB_EXCEPTION(err, m_database);
				}

				::bson_iter_t it;
				if(!::bson_iter_init_find(&it, reply_bt, "cursor")){
					LOG_POSEIDON_DEBUG("No cursor returned from MongoDB server.");
				} else {
					DEBUG_THROW_ASSERT(::bson_iter_type(&it) == BSON_TYPE_DOCUMENT);
					boost::uint32_t size;
					const boost::uint8_t *data;
					::bson_iter_document(&it, &size, &data);
					::bson_t cursor_storage;
					success = ::bson_init_static(&cursor_storage, data, size);
					DEBUG_THROW_ASSERT(success);
					const UniqueHandle<BsonCloser> cursor_guard(&cursor_storage);
					const AUTO(cursor_bt, cursor_guard.get());

					if(::bson_iter_init_find(&it, cursor_bt, "id")){
						DEBUG_THROW_ASSERT(::bson_iter_type(&it) == BSON_TYPE_INT64);
						m_cursor_id = ::bson_iter_int64(&it);
					}
					if(::bson_iter_init_find(&it, cursor_bt, "ns")){
						DEBUG_THROW_ASSERT(::bson_iter_type(&it) == BSON_TYPE_UTF8);
						m_cursor_ns = ::bson_iter_utf8(&it, NULLPTR);
					}
					if(::bson_iter_init_find(&it, cursor_bt, "firstBatch")){
						DEBUG_THROW_ASSERT(::bson_iter_type(&it) == BSON_TYPE_ARRAY);
						::bson_iter_array(&it, &size, &data);
						m_batch_guard.reset(::bson_new_from_data(data, size));
						DEBUG_THROW_ASSERT(m_batch_guard);
						const AUTO(batch_bt, m_batch_guard.get());
						success = ::bson_iter_init(&m_batch_it, batch_bt);
						DEBUG_THROW_ASSERT(success);
					}
				}
			}
			void do_discard_result() NOEXCEPT {
				m_cursor_id = 0;
				m_cursor_ns.clear();
				m_batch_guard.reset();
				m_element_guard.reset();
			}

			bool do_fetch_next(){
				while(m_batch_guard && !::bson_iter_next(&m_batch_it)){
					m_batch_guard.reset();

					if(m_cursor_id != 0){
						LOG_POSEIDON_DEBUG("Fetching more data: cursor_id = ", m_cursor_id);

						const AUTO(query_bt, ::bson_sized_new(256));
						DEBUG_THROW_ASSERT(query_bt);
						const UniqueHandle<BsonCloser> query_guard(query_bt);
						bool success = ::bson_append_int64(query_bt, "getMore", -1, m_cursor_id);
						DEBUG_THROW_ASSERT(success);
						const AUTO(db_len, std::strlen(m_database));
						DEBUG_THROW_ASSERT(m_cursor_ns.compare(0, db_len, m_database.get()) == 0);
						DEBUG_THROW_ASSERT(m_cursor_ns.at(db_len) == '.');
						success = ::bson_append_utf8(query_bt, "collection", -1, m_cursor_ns.c_str() + db_len + 1, -1);
						DEBUG_THROW_ASSERT(success);

						do_discard_result();

						::bson_t reply_storage;
						::bson_error_t err;
						success = ::mongoc_client_command_simple(m_client.get(), m_database.get(), query_bt, NULLPTR, &reply_storage, &err);
						// `reply` is always set.
						const UniqueHandle<BsonCloser> reply_guard(&reply_storage);
						const AUTO(reply_bt, reply_guard.get());
						if(!success){
							DEBUG_THROW_MONGODB_EXCEPTION(err, m_database);
						}

						::bson_iter_t it;
						if(!::bson_iter_init_find(&it, reply_bt, "cursor")){
							LOG_POSEIDON_DEBUG("No cursor returned from MongoDB server.");
						} else {
							DEBUG_THROW_ASSERT(::bson_iter_type(&it) == BSON_TYPE_DOCUMENT);
							boost::uint32_t size;
							const boost::uint8_t *data;
							::bson_iter_document(&it, &size, &data);
							::bson_t cursor_storage;
							success = ::bson_init_static(&cursor_storage, data, size);
							DEBUG_THROW_ASSERT(success);
							const UniqueHandle<BsonCloser> cursor_guard(&cursor_storage);
							const AUTO(cursor_bt, cursor_guard.get());

							if(::bson_iter_init_find(&it, cursor_bt, "id")){
								DEBUG_THROW_ASSERT(::bson_iter_type(&it) == BSON_TYPE_INT64);
								m_cursor_id = ::bson_iter_int64(&it);
							}
							if(::bson_iter_init_find(&it, cursor_bt, "ns")){
								DEBUG_THROW_ASSERT(::bson_iter_type(&it) == BSON_TYPE_UTF8);
								m_cursor_ns = ::bson_iter_utf8(&it, NULLPTR);
							}
							if(::bson_iter_init_find(&it, cursor_bt, "nextBatch")){
								DEBUG_THROW_ASSERT(::bson_iter_type(&it) == BSON_TYPE_ARRAY);
								::bson_iter_array(&it, &size, &data);
								m_batch_guard.reset(::bson_new_from_data(data, size));
								DEBUG_THROW_ASSERT(m_batch_guard);
								const AUTO(batch_bt, m_batch_guard.get());
								success = ::bson_iter_init(&m_batch_it, batch_bt);
								DEBUG_THROW_ASSERT(success);
							}
						}
					}
				}
				if(!m_batch_guard){
					LOG_POSEIDON_DEBUG("No more data.");
					return false;
				}
				DEBUG_THROW_ASSERT(::bson_iter_type(&m_batch_it) == BSON_TYPE_DOCUMENT);
				boost::uint32_t size;
				const boost::uint8_t *data;
				::bson_iter_document(&m_batch_it, &size, &data);
				if(!::bson_init_static(&m_element_storage, data, size)){
					LOG_POSEIDON_ERROR("::bson_init_static() failed!");
					DEBUG_THROW(ProtocolException, sslit("::bson_init_static() failed"), -1);
				}
				m_element_guard.reset(&m_element_storage);
				return true;
			}

			bool do_get_boolean(const char *name) const {
				::bson_iter_t it;
				if(!find_bson_element_and_check_type(it, name, BSON_TYPE_BOOL)){
					return VAL_INIT;
				}
				const bool value = ::bson_iter_bool(&it);
				return value;
			}
			boost::int64_t do_get_signed(const char *name) const {
				::bson_iter_t it;
				if(!find_bson_element_and_check_type(it, name, BSON_TYPE_INT64)){
					return VAL_INIT;
				}
				const boost::int64_t value = ::bson_iter_int64(&it);
				return value;
			}
			boost::uint64_t do_get_unsigned(const char *name) const {
				::bson_iter_t it;
				if(!find_bson_element_and_check_type(it, name, BSON_TYPE_INT64)){
					return VAL_INIT;
				}
				const boost::int64_t shifted = ::bson_iter_int64(&it);
				return static_cast<boost::uint64_t>(shifted) + (1ull << 63);
			}
			double do_get_double(const char *name) const {
				::bson_iter_t it;
				if(!find_bson_element_and_check_type(it, name, BSON_TYPE_DOUBLE)){
					return VAL_INIT;
				}
				const double value = ::bson_iter_double(&it);
				return value;
			}
			std::string do_get_string(const char *name) const {
				::bson_iter_t it;
				if(!find_bson_element_and_check_type(it, name, BSON_TYPE_UTF8)){
					return VAL_INIT;
				}
				boost::uint32_t len;
				const char *const str = ::bson_iter_utf8(&it, &len);
				return std::string(str, len);
			}
			boost::uint64_t do_get_datetime(const char *name) const {
				::bson_iter_t it;
				if(!find_bson_element_and_check_type(it, name, BSON_TYPE_UTF8)){
					return VAL_INIT;
				}
				const char *const str = ::bson_iter_utf8(&it, NULLPTR);
				return scan_time(str);
			}
			Uuid do_get_uuid(const char *name) const {
				::bson_iter_t it;
				if(!find_bson_element_and_check_type(it, name, BSON_TYPE_UTF8)){
					return VAL_INIT;
				}
				boost::uint32_t len;
				const char *const str = ::bson_iter_utf8(&it, &len);
				if(len != 36){
					DEBUG_THROW(BasicException, sslit("Unexpected UUID string length"));
				}
				return Uuid(reinterpret_cast<const char (&)[36]>(*str));
			}
			std::basic_string<unsigned char> do_get_blob(const char *name) const {
				::bson_iter_t it;
				if(!find_bson_element_and_check_type(it, name, BSON_TYPE_BINARY)){
					return VAL_INIT;
				}
				boost::uint32_t len;
				const boost::uint8_t *data;
				::bson_iter_binary(&it, NULLPTR, &len, &data);
				return std::basic_string<unsigned char>(data, len);
			}
		};
	}

	boost::shared_ptr<Connection> Connection::create(const char *server_addr, unsigned server_port,
		const char *user_name, const char *password, const char *auth_database, bool use_ssl, const char *database)
	{
		return boost::make_shared<DelegatedConnection>(server_addr, server_port,
			user_name, password, auth_database, use_ssl, database);
	}

	Connection::~Connection(){ }

	void Connection::execute_bson(const BsonBuilder &bson){
		static_cast<DelegatedConnection &>(*this).do_execute_bson(bson);
	}
	void Connection::discard_result() NOEXCEPT {
		static_cast<DelegatedConnection &>(*this).do_discard_result();
	}

	bool Connection::fetch_next(){
		return static_cast<DelegatedConnection &>(*this).do_fetch_next();
	}

	bool Connection::get_boolean(const char *name) const {
		return static_cast<const DelegatedConnection &>(*this).do_get_boolean(name);
	}
	boost::int64_t Connection::get_signed(const char *name) const {
		return static_cast<const DelegatedConnection &>(*this).do_get_signed(name);
	}
	boost::uint64_t Connection::get_unsigned(const char *name) const {
		return static_cast<const DelegatedConnection &>(*this).do_get_unsigned(name);
	}
	double Connection::get_double(const char *name) const {
		return static_cast<const DelegatedConnection &>(*this).do_get_double(name);
	}
	std::string Connection::get_string(const char *name) const {
		return static_cast<const DelegatedConnection &>(*this).do_get_string(name);
	}
	boost::uint64_t Connection::get_datetime(const char *name) const {
		return static_cast<const DelegatedConnection &>(*this).do_get_datetime(name);
	}
	Uuid Connection::get_uuid(const char *name) const {
		return static_cast<const DelegatedConnection &>(*this).do_get_uuid(name);
	}
	std::basic_string<unsigned char> Connection::get_blob(const char *name) const {
		return static_cast<const DelegatedConnection &>(*this).do_get_blob(name);
	}
}

}
