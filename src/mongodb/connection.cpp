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
namespace Mongo_db {

namespace {
	void init_mongoc(){
		::mongoc_init();

		std::atexit(&::mongoc_cleanup);
	}

	::pthread_once_t g_mongoc_once = PTHREAD_ONCE_INIT;

	void init_mongoc_once(){
		DEBUG_THROW_ASSERT(::pthread_once(&g_mongoc_once, &init_mongoc) == 0);
	}

	struct Uri_closer {
		CONSTEXPR ::mongoc_uri_t *operator()() const NOEXCEPT {
			return NULLPTR;
		}
		void operator()(::mongoc_uri_t *uri) const NOEXCEPT {
			::mongoc_uri_destroy(uri);
		}
	};
	struct Client_closer {
		CONSTEXPR ::mongoc_client_t *operator()() const NOEXCEPT {
			return NULLPTR;
		}
		void operator()(::mongoc_client_t *client) const NOEXCEPT {
			::mongoc_client_destroy(client);
		}
	};

	struct Bson_closer {
		CONSTEXPR ::bson_t *operator()() const NOEXCEPT {
			return NULLPTR;
		}
		void operator()(::bson_t *bson) const NOEXCEPT {
			::bson_destroy(bson);
		}
	};

	class Delegated_connection : public Connection {
	private:
		Shared_nts m_database;
		Unique_handle<Client_closer> m_client;

		boost::int64_t m_cursor_id;
		std::string m_cursor_ns;
		Unique_handle<Bson_closer> m_batch_guard;
		::bson_iter_t m_batch_it;
		::bson_t m_element_storage;
		Unique_handle<Bson_closer> m_element_guard;

	public:
		Delegated_connection(const char *server_addr, boost::uint16_t server_port, const char *user_name, const char *password, const char *auth_database, bool use_ssl, const char *database)
			: m_database(database)
			, m_cursor_id(0), m_cursor_ns()
		{
			PROFILE_ME;

			init_mongoc_once();

			Unique_handle<Uri_closer> uri;
			DEBUG_THROW_UNLESS(uri.reset(::mongoc_uri_new_for_host_port(server_addr, server_port)), Basic_exception, sslit("::mongoc_uri_new_for_host_port() failed"));
			DEBUG_THROW_UNLESS(::mongoc_uri_set_username(uri.get(), user_name), Basic_exception, sslit("::mongoc_uri_set_username() failed"));
			DEBUG_THROW_UNLESS(::mongoc_uri_set_password(uri.get(), password), Basic_exception, sslit("::mongoc_uri_set_password() failed"));
			DEBUG_THROW_UNLESS(::mongoc_uri_set_database(uri.get(), auth_database), Basic_exception, sslit("::mongoc_uri_set_database() failed"));
			DEBUG_THROW_UNLESS(::mongoc_uri_set_option_as_bool(uri.get(), "ssl", use_ssl), Basic_exception, sslit("::mongoc_uri_set_option_as_bool() failed"));
			DEBUG_THROW_UNLESS(m_client.reset(::mongoc_client_new_from_uri(uri.get())), Basic_exception, sslit("::mongoc_client_new_from_uri() failed"));
		}

	private:
		bool parse_reply_cursor(const ::bson_t *reply_bt, const char *batch_id){
			PROFILE_ME;

			::bson_iter_t it;
			if(!::bson_iter_init_find(&it, reply_bt, "cursor")){
				LOG_POSEIDON_DEBUG("No cursor was returned from MongoDB server.");
				return false;
			}
			DEBUG_THROW_ASSERT(::bson_iter_type(&it) == BSON_TYPE_DOCUMENT);
			boost::uint32_t size;
			const boost::uint8_t *data;
			::bson_iter_document(&it, &size, &data);
			::bson_t cursor_storage;
			DEBUG_THROW_ASSERT(::bson_init_static(&cursor_storage, data, size));
			const Unique_handle<Bson_closer> cursor_guard(&cursor_storage);
			const AUTO(cursor_bt, cursor_guard.get());

			if(::bson_iter_init_find(&it, cursor_bt, "id")){
				DEBUG_THROW_ASSERT(::bson_iter_type(&it) == BSON_TYPE_INT64);
				m_cursor_id = ::bson_iter_int64(&it);
				LOG_POSEIDON_TRACE("Parsing MongoDB reply cursor: cursor_id = ", m_cursor_id);
			}
			if(::bson_iter_init_find(&it, cursor_bt, "ns")){
				DEBUG_THROW_ASSERT(::bson_iter_type(&it) == BSON_TYPE_UTF8);
				m_cursor_ns = ::bson_iter_utf8(&it, NULLPTR);
				LOG_POSEIDON_TRACE("Parsing MongoDB reply cursor: cursor_ns = ", m_cursor_ns);
			}
			if(::bson_iter_init_find(&it, cursor_bt, batch_id)){
				DEBUG_THROW_ASSERT(::bson_iter_type(&it) == BSON_TYPE_ARRAY);
				::bson_iter_array(&it, &size, &data);
				DEBUG_THROW_ASSERT(m_batch_guard.reset(::bson_new_from_data(data, size)));
				const AUTO(batch_bt, m_batch_guard.get());
				DEBUG_THROW_ASSERT(::bson_iter_init(&m_batch_it, batch_bt));
			}
			return true;
		}

		::bson_type_t find_bson_element_and_check(::bson_iter_t &it, const char *name) const {
			PROFILE_ME;

			if(!m_element_guard){
				LOG_POSEIDON_WARNING("No more results available.");
				return BSON_TYPE_EOD;
			}
			const AUTO(element, m_element_guard.get());
			if(!::bson_iter_init_find(&it, element, name)){
				LOG_POSEIDON_WARNING("Field not found: name = ", name);
				return BSON_TYPE_EOD;
			}
			const AUTO(type, ::bson_iter_type(&it));
			if((type == BSON_TYPE_UNDEFINED) || (type == BSON_TYPE_NULL)){
				LOG_POSEIDON_DEBUG("Field is `undefined` or `null`: name = ", name);
				return BSON_TYPE_EOD;
			}
			return type;
		}

	public:
		void execute_bson(const Bson_builder &bson) FINAL {
			PROFILE_ME;

			const AUTO(query_data, bson.build(false));
			::bson_t query_storage;
			DEBUG_THROW_ASSERT(::bson_init_static(&query_storage, reinterpret_cast<const boost::uint8_t *>(query_data.data()), query_data.size()));
			const Unique_handle<Bson_closer> query_guard(&query_storage);
			const AUTO(query_bt, query_guard.get());

			discard_result();

			LOG_POSEIDON_DEBUG("Sending query to MongoDB server: ", bson.build_json());
			::bson_t reply_storage;
			::bson_error_t err;
			bool success = ::mongoc_client_command_simple(m_client.get(), m_database.get(), query_bt, NULLPTR, &reply_storage, &err);
			// `reply` is always set.
			const Unique_handle<Bson_closer> reply_guard(&reply_storage);
			const AUTO(reply_bt, reply_guard.get());
			DEBUG_THROW_UNLESS(success, Exception, m_database, err.code, Shared_nts(err.message));
			parse_reply_cursor(reply_bt, "firstBatch");
		}
		void discard_result() NOEXCEPT FINAL {
			PROFILE_ME;

			m_cursor_id = 0;
			m_cursor_ns.clear();
			m_batch_guard.reset();
			m_element_guard.reset();
		}

		bool fetch_document() FINAL {
			PROFILE_ME;

			for(;;){
				if(m_batch_guard){
					if(::bson_iter_next(&m_batch_it)){
						break;
					}
					m_batch_guard.reset();
				}
				if(m_cursor_id == 0){
					LOG_POSEIDON_DEBUG("No more data.");
					return false;
				}
				LOG_POSEIDON_DEBUG("Issuing a `getMore` request: cursor_id = ", m_cursor_id);

				Unique_handle<Bson_closer> query_guard;
				DEBUG_THROW_ASSERT(query_guard.reset(::bson_sized_new(1024)));
				const AUTO(query_bt, query_guard.get());
				DEBUG_THROW_ASSERT(::bson_append_int64(query_bt, "getMore", -1, m_cursor_id));
				const AUTO(database_len, std::strlen(m_database));
				DEBUG_THROW_ASSERT(m_cursor_ns.compare(0, database_len, m_database.get()) == 0);
				DEBUG_THROW_ASSERT(m_cursor_ns.at(database_len) == '.');
				DEBUG_THROW_ASSERT(::bson_append_utf8(query_bt, "collection", -1, m_cursor_ns.c_str() + database_len + 1, -1));

				discard_result();

				::bson_t reply_storage;
				::bson_error_t err;
				bool success = ::mongoc_client_command_simple(m_client.get(), m_database.get(), query_bt, NULLPTR, &reply_storage, &err);
				// `reply` is always set.
				const Unique_handle<Bson_closer> reply_guard(&reply_storage);
				const AUTO(reply_bt, reply_guard.get());
				DEBUG_THROW_UNLESS(success, Exception, m_database, err.code, Shared_nts(err.message));
				parse_reply_cursor(reply_bt, "nextBatch");
			}
			DEBUG_THROW_ASSERT(::bson_iter_type(&m_batch_it) == BSON_TYPE_DOCUMENT);
			boost::uint32_t size;
			const boost::uint8_t *data;
			::bson_iter_document(&m_batch_it, &size, &data);
			DEBUG_THROW_UNLESS(::bson_init_static(&m_element_storage, data, size), Basic_exception, sslit("::bson_init_static() failed"));
			m_element_guard.reset(&m_element_storage);
			return true;
		}

		bool get_boolean(const char *name) const FINAL {
			PROFILE_ME;

			bool value = false;
			::bson_iter_t it;
			switch(find_bson_element_and_check(it, name)){
			case BSON_TYPE_EOD:
				break;
			case BSON_TYPE_BOOL:
				value = ::bson_iter_bool(&it);
				break;
			case BSON_TYPE_INT32:
				value = ::bson_iter_int32(&it) != 0;
				break;
			case BSON_TYPE_INT64:
				value = ::bson_iter_int64(&it) != 0;
				break;
			case BSON_TYPE_DOUBLE:
				value = ::bson_iter_double(&it) != 0;
				break;
			case BSON_TYPE_UTF8: {
				boost::uint32_t size;
				const char *data;
				data = ::bson_iter_utf8(&it, &size);
				value = (size != 0) && (std::strcmp(data, "0") != 0);
				break; }
			case BSON_TYPE_BINARY: {
				boost::uint32_t size;
				const boost::uint8_t *data;
				::bson_iter_binary(&it, NULLPTR, &size, &data);
				value = size != 0;
				break; }
			case BSON_TYPE_DOCUMENT:
			case BSON_TYPE_ARRAY:
				value = true;
				break;
			default:
				LOG_POSEIDON_ERROR("BSON data type not handled: name = ", name, ", type = ", ::bson_iter_type(&it));
				DEBUG_THROW(Basic_exception, sslit("Unexpected BSON data type"));
			}
			return value;
		}
		boost::int64_t get_signed(const char *name) const FINAL {
			PROFILE_ME;

			boost::int64_t value = 0;
			::bson_iter_t it;
			switch(find_bson_element_and_check(it, name)){
			case BSON_TYPE_EOD:
				break;
			case BSON_TYPE_BOOL:
				value = ::bson_iter_bool(&it);
				break;
			case BSON_TYPE_INT32:
				value = ::bson_iter_int32(&it);
				break;
			case BSON_TYPE_INT64:
				value = ::bson_iter_int64(&it);
				break;
			case BSON_TYPE_DOUBLE:
				value = boost::numeric_cast<boost::int64_t>(::bson_iter_double(&it));
				break;
			case BSON_TYPE_UTF8: {
				boost::uint32_t size;
				const char *data;
				data = ::bson_iter_utf8(&it, &size);
				char *eptr;
				value = ::strtoll(data, &eptr, 0);
				DEBUG_THROW_UNLESS(*eptr == 0, Basic_exception, sslit("Could not convert field data to `long long`"));
				break; }
			default:
				LOG_POSEIDON_ERROR("BSON data type not handled: name = ", name, ", type = ", ::bson_iter_type(&it));
				DEBUG_THROW(Basic_exception, sslit("Unexpected BSON data type"));
			}
			return value;
		}
		boost::uint64_t get_unsigned(const char *name) const FINAL {
			PROFILE_ME;

			boost::uint64_t value = 0;
			::bson_iter_t it;
			switch(find_bson_element_and_check(it, name)){
			case BSON_TYPE_EOD:
				break;
			case BSON_TYPE_BOOL:
				value = ::bson_iter_bool(&it);
				break;
			case BSON_TYPE_INT32:
				value = boost::numeric_cast<boost::uint32_t>(::bson_iter_int32(&it));
				break;
			case BSON_TYPE_INT64:
				value = boost::numeric_cast<boost::uint64_t>(::bson_iter_int64(&it));
				break;
			case BSON_TYPE_DOUBLE:
				value = boost::numeric_cast<boost::uint64_t>(::bson_iter_double(&it));
				break;
			case BSON_TYPE_UTF8: {
				boost::uint32_t size;
				const char *data;
				data = ::bson_iter_utf8(&it, &size);
				char *eptr;
				value = ::strtoull(data, &eptr, 0);
				DEBUG_THROW_UNLESS(*eptr == 0, Basic_exception, sslit("Could not convert field data to `unsigned long long`"));
				break; }
			default:
				LOG_POSEIDON_ERROR("BSON data type not handled: name = ", name, ", type = ", ::bson_iter_type(&it));
				DEBUG_THROW(Basic_exception, sslit("Unexpected BSON data type"));
			}
			return value;
		}
		double get_double(const char *name) const FINAL {
			PROFILE_ME;

			double value = 0;
			::bson_iter_t it;
			switch(find_bson_element_and_check(it, name)){
			case BSON_TYPE_EOD:
				break;
			case BSON_TYPE_BOOL:
				value = ::bson_iter_bool(&it);
				break;
			case BSON_TYPE_INT32:
				value = ::bson_iter_int32(&it);
				break;
			case BSON_TYPE_INT64:
				value = boost::numeric_cast<double>(::bson_iter_int64(&it));
				break;
			case BSON_TYPE_DOUBLE:
				value = ::bson_iter_double(&it);
				break;
			case BSON_TYPE_UTF8: {
				boost::uint32_t size;
				const char *data;
				data = ::bson_iter_utf8(&it, &size);
				char *eptr;
				value = ::strtod(data, &eptr);
				DEBUG_THROW_UNLESS(*eptr == 0, Basic_exception, sslit("Could not convert field data to `double`"));
				break; }
			default:
				LOG_POSEIDON_ERROR("BSON data type not handled: name = ", name, ", type = ", ::bson_iter_type(&it));
				DEBUG_THROW(Basic_exception, sslit("Unexpected BSON data type"));
			}
			return value;
		}
		std::string get_string(const char *name) const FINAL {
			PROFILE_ME;

			std::string value;
			::bson_iter_t it;
			switch(find_bson_element_and_check(it, name)){
			case BSON_TYPE_EOD:
				break;
			case BSON_TYPE_BOOL:
				value = ::bson_iter_bool(&it) ? "true" : "false";
				break;
			case BSON_TYPE_INT32:
				value = boost::lexical_cast<std::string>(::bson_iter_int32(&it));
				break;
			case BSON_TYPE_INT64:
				value = boost::lexical_cast<std::string>(::bson_iter_int64(&it));
				break;
			case BSON_TYPE_DOUBLE:
				value = boost::lexical_cast<std::string>(::bson_iter_double(&it));
				break;
			case BSON_TYPE_UTF8: {
				boost::uint32_t size;
				const char *data;
				data = ::bson_iter_utf8(&it, &size);
				value.assign(data, size);
				break; }
			case BSON_TYPE_BINARY: {
				boost::uint32_t size;
				const boost::uint8_t *data;
				::bson_iter_binary(&it, NULLPTR, &size, &data);
				value.assign(reinterpret_cast<const char *>(data), size);
				break; }
			default:
				LOG_POSEIDON_ERROR("BSON data type not handled: name = ", name, ", type = ", ::bson_iter_type(&it));
				DEBUG_THROW(Basic_exception, sslit("Unexpected BSON data type"));
			}
			return value;
		}
		boost::uint64_t get_datetime(const char *name) const FINAL {
			PROFILE_ME;

			boost::uint64_t value = 0;
			::bson_iter_t it;
			switch(find_bson_element_and_check(it, name)){
			case BSON_TYPE_EOD:
				break;
			case BSON_TYPE_UTF8: {
				boost::uint32_t size;
				const char *data;
				data = ::bson_iter_utf8(&it, &size);
				value = scan_time(data);
				break; }
			default:
				LOG_POSEIDON_ERROR("BSON data type not handled: name = ", name, ", type = ", ::bson_iter_type(&it));
				DEBUG_THROW(Basic_exception, sslit("Unexpected BSON data type"));
			}
			return value;
		}
		Uuid get_uuid(const char *name) const FINAL {
			PROFILE_ME;

			Uuid value;
			::bson_iter_t it;
			switch(find_bson_element_and_check(it, name)){
			case BSON_TYPE_EOD:
				break;
			case BSON_TYPE_UTF8: {
				boost::uint32_t size;
				const char *data;
				data = ::bson_iter_utf8(&it, &size);
				DEBUG_THROW_UNLESS(size == 36, Basic_exception, sslit("Invalid UUID string length"));
				value.from_string(*reinterpret_cast<const char (*)[36]>(data));
				break; }
			default:
				LOG_POSEIDON_ERROR("BSON data type not handled: name = ", name, ", type = ", ::bson_iter_type(&it));
				DEBUG_THROW(Basic_exception, sslit("Unexpected BSON data type"));
			}
			return value;
		}
		std::basic_string<unsigned char> get_blob(const char *name) const FINAL {
			PROFILE_ME;

			std::basic_string<unsigned char> value;
			::bson_iter_t it;
			switch(find_bson_element_and_check(it, name)){
			case BSON_TYPE_EOD:
				break;
			case BSON_TYPE_UTF8: {
				boost::uint32_t size;
				const char *data;
				data = ::bson_iter_utf8(&it, &size);
				value.assign(reinterpret_cast<const boost::uint8_t *>(data), size);
				break; }
			case BSON_TYPE_BINARY: {
				boost::uint32_t size;
				const boost::uint8_t *data;
				::bson_iter_binary(&it, NULLPTR, &size, &data);
				value.assign(data, size);
				break; }
			default:
				LOG_POSEIDON_ERROR("BSON data type not handled: name = ", name, ", type = ", ::bson_iter_type(&it));
				DEBUG_THROW(Basic_exception, sslit("Unexpected BSON data type"));
			}
			return value;
		}
	};
}

boost::shared_ptr<Connection> Connection::create(const char *server_addr, boost::uint16_t server_port, const char *user_name, const char *password, const char *auth_database, bool use_ssl, const char *database){
	return boost::make_shared<Delegated_connection>(server_addr, server_port, user_name, password, auth_database, use_ssl, database);
}

Connection::~Connection(){
	//
}

}
}
