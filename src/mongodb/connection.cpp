// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "connection.hpp"
#include "exception.hpp"
#include "bson_builder.hpp"
#include "../raii.hpp"
#include "../log.hpp"
#include "../profiler.hpp"
#include "../time.hpp"
#include <cstdlib>
#include <libbson-1.0/bson.h>
#include <libmongoc-1.0/mongoc.h>

namespace Poseidon {
namespace MongoDb {

namespace {
	void init_mongoc(){
		::mongoc_init();

		std::atexit(&::mongoc_cleanup);
	}

	::pthread_once_t g_mongoc_once = PTHREAD_ONCE_INIT;

	void init_mongoc_once(){
		DEBUG_THROW_ASSERT(::pthread_once(&g_mongoc_once, &init_mongoc) == 0);
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

	class DelegatedConnection : public Connection {
	private:
		SharedNts m_database;
		UniqueHandle<ClientCloser> m_client;

		boost::int64_t m_cursor_id;
		std::string m_cursor_ns;
		UniqueHandle<BsonCloser> m_batch_guard;
		::bson_iter_t m_batch_it;
		::bson_t m_element_storage;
		UniqueHandle<BsonCloser> m_element_guard;

	public:
		DelegatedConnection(const char *server_addr, boost::uint16_t server_port, const char *user_name, const char *password, const char *auth_database, bool use_ssl, const char *database)
			: m_database(database)
			, m_cursor_id(0), m_cursor_ns()
		{
			PROFILE_ME;

			init_mongoc_once();

			UniqueHandle<UriCloser> uri;
			DEBUG_THROW_UNLESS(uri.reset(::mongoc_uri_new_for_host_port(server_addr, server_port)), BasicException, sslit("::mongoc_uri_new_for_host_port() failed"));
			DEBUG_THROW_UNLESS(::mongoc_uri_set_username(uri.get(), user_name), BasicException, sslit("::mongoc_uri_set_username() failed"));
			DEBUG_THROW_UNLESS(::mongoc_uri_set_password(uri.get(), password), BasicException, sslit("::mongoc_uri_set_password() failed"));
			DEBUG_THROW_UNLESS(::mongoc_uri_set_database(uri.get(), auth_database), BasicException, sslit("::mongoc_uri_set_database() failed"));
			DEBUG_THROW_UNLESS(::mongoc_uri_set_option_as_bool(uri.get(), "ssl", use_ssl), BasicException, sslit("::mongoc_uri_set_option_as_bool() failed"));
			DEBUG_THROW_UNLESS(m_client.reset(::mongoc_client_new_from_uri(uri.get())), BasicException, sslit("::mongoc_client_new_from_uri() failed"));
		}

	private:
		bool find_bson_element_and_check_type(::bson_iter_t &it, const char *name, ::bson_type_t type_expecting) const {
			PROFILE_ME;

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
			DEBUG_THROW_UNLESS(type == type_expecting, BasicException, sslit("BSON type mismatch"));
			return true;
		}

	public:
		void execute_bson(const BsonBuilder &bson) FINAL {
			PROFILE_ME;

			const AUTO(query_data, bson.build(false));
			::bson_t query_storage;
			DEBUG_THROW_ASSERT(::bson_init_static(&query_storage, reinterpret_cast<const boost::uint8_t *>(query_data.data()), query_data.size()));
			const UniqueHandle<BsonCloser> query_guard(&query_storage);
			const AUTO(query_bt, query_guard.get());

			discard_result();

			LOG_POSEIDON_DEBUG("Sending query to MongoDB server: ", bson.build_json());
			::bson_t reply_storage;
			::bson_error_t err;
			bool success = ::mongoc_client_command_simple(m_client.get(), m_database.get(), query_bt, NULLPTR, &reply_storage, &err);
			// `reply` is always set.
			const UniqueHandle<BsonCloser> reply_guard(&reply_storage);
			const AUTO(reply_bt, reply_guard.get());
			DEBUG_THROW_UNLESS(success, Exception, m_database, err.code, SharedNts(err.message));

			::bson_iter_t it;
			if(::bson_iter_init_find(&it, reply_bt, "cursor")){
				DEBUG_THROW_ASSERT(::bson_iter_type(&it) == BSON_TYPE_DOCUMENT);
				boost::uint32_t size;
				const boost::uint8_t *data;
				::bson_iter_document(&it, &size, &data);
				::bson_t cursor_storage;
				DEBUG_THROW_ASSERT(::bson_init_static(&cursor_storage, data, size));
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
					DEBUG_THROW_ASSERT(::bson_iter_init(&m_batch_it, batch_bt));
				}
			} else {
				LOG_POSEIDON_DEBUG("No cursor was returned from MongoDB server.");
			}
		}
		void discard_result() NOEXCEPT FINAL {
			PROFILE_ME;

			m_cursor_id = 0;
			m_cursor_ns.clear();
			m_batch_guard.reset();
			m_element_guard.reset();
		}

		bool fetch_next() FINAL {
			PROFILE_ME;

			while(m_batch_guard && !::bson_iter_next(&m_batch_it)){
				m_batch_guard.reset();

				if(m_cursor_id != 0){
					LOG_POSEIDON_DEBUG("Fetching more data: cursor_id = ", m_cursor_id);

					const AUTO(query_bt, ::bson_sized_new(256));
					DEBUG_THROW_ASSERT(query_bt);
					const UniqueHandle<BsonCloser> query_guard(query_bt);
					DEBUG_THROW_ASSERT(::bson_append_int64(query_bt, "getMore", -1, m_cursor_id));
					const AUTO(db_len, std::strlen(m_database));
					DEBUG_THROW_ASSERT(m_cursor_ns.compare(0, db_len, m_database.get()) == 0);
					DEBUG_THROW_ASSERT(m_cursor_ns.at(db_len) == '.');
					DEBUG_THROW_ASSERT(::bson_append_utf8(query_bt, "collection", -1, m_cursor_ns.c_str() + db_len + 1, -1));

					discard_result();

					::bson_t reply_storage;
					::bson_error_t err;
					bool success = ::mongoc_client_command_simple(m_client.get(), m_database.get(), query_bt, NULLPTR, &reply_storage, &err);
					// `reply` is always set.
					const UniqueHandle<BsonCloser> reply_guard(&reply_storage);
					const AUTO(reply_bt, reply_guard.get());
					DEBUG_THROW_UNLESS(success, Exception, m_database, err.code, SharedNts(err.message));

					::bson_iter_t it;
					if(::bson_iter_init_find(&it, reply_bt, "cursor")){
						DEBUG_THROW_ASSERT(::bson_iter_type(&it) == BSON_TYPE_DOCUMENT);
						boost::uint32_t size;
						const boost::uint8_t *data;
						::bson_iter_document(&it, &size, &data);
						::bson_t cursor_storage;
						DEBUG_THROW_ASSERT(::bson_init_static(&cursor_storage, data, size));
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
							DEBUG_THROW_ASSERT(::bson_iter_init(&m_batch_it, batch_bt));
						}
					} else {
						LOG_POSEIDON_DEBUG("No cursor returned from MongoDB server.");
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
			DEBUG_THROW_UNLESS(::bson_init_static(&m_element_storage, data, size), BasicException, sslit("::bson_init_static() failed"));
			m_element_guard.reset(&m_element_storage);
			return true;
		}

		bool get_boolean(const char *name) const FINAL {
			PROFILE_ME;

			::bson_iter_t it;
			if(!find_bson_element_and_check_type(it, name, BSON_TYPE_BOOL)){
				return VAL_INIT;
			}
			const bool value = ::bson_iter_bool(&it);
			return value;
		}
		boost::int64_t get_signed(const char *name) const FINAL {
			PROFILE_ME;

			::bson_iter_t it;
			if(!find_bson_element_and_check_type(it, name, BSON_TYPE_INT64)){
				return VAL_INIT;
			}
			const boost::int64_t value = ::bson_iter_int64(&it);
			return value;
		}
		boost::uint64_t get_unsigned(const char *name) const FINAL {
			PROFILE_ME;

			::bson_iter_t it;
			if(!find_bson_element_and_check_type(it, name, BSON_TYPE_INT64)){
				return VAL_INIT;
			}
			const boost::int64_t shifted = ::bson_iter_int64(&it);
			return static_cast<boost::uint64_t>(shifted) + (1ull << 63);
		}
		double get_double(const char *name) const FINAL {
			PROFILE_ME;

			::bson_iter_t it;
			if(!find_bson_element_and_check_type(it, name, BSON_TYPE_DOUBLE)){
				return VAL_INIT;
			}
			const double value = ::bson_iter_double(&it);
			return value;
		}
		std::string get_string(const char *name) const FINAL {
			PROFILE_ME;

			::bson_iter_t it;
			if(!find_bson_element_and_check_type(it, name, BSON_TYPE_UTF8)){
				return VAL_INIT;
			}
			boost::uint32_t len;
			const char *const str = ::bson_iter_utf8(&it, &len);
			return std::string(str, len);
		}
		boost::uint64_t get_datetime(const char *name) const FINAL {
			PROFILE_ME;

			::bson_iter_t it;
			if(!find_bson_element_and_check_type(it, name, BSON_TYPE_UTF8)){
				return VAL_INIT;
			}
			const char *const str = ::bson_iter_utf8(&it, NULLPTR);
			return scan_time(str);
		}
		Uuid get_uuid(const char *name) const FINAL {
			PROFILE_ME;

			::bson_iter_t it;
			if(!find_bson_element_and_check_type(it, name, BSON_TYPE_UTF8)){
				return VAL_INIT;
			}
			boost::uint32_t len;
			const char *const str = ::bson_iter_utf8(&it, &len);
			DEBUG_THROW_UNLESS(len == 36, BasicException, sslit("Unexpected UUID string length"));
			return Uuid(reinterpret_cast<const char (&)[36]>(*str));
		}
		std::basic_string<unsigned char> get_blob(const char *name) const FINAL {
			PROFILE_ME;

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

boost::shared_ptr<Connection> Connection::create(const char *server_addr, boost::uint16_t server_port, const char *user_name, const char *password, const char *auth_database, bool use_ssl, const char *database){
	return boost::make_shared<DelegatedConnection>(server_addr, server_port, user_name, password, auth_database, use_ssl, database);
}

Connection::~Connection(){
	//
}

}
}
