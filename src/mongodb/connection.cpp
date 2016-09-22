// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2016, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "connection.hpp"
#include "exception.hpp"
#include "oid.hpp"
#include "bson_builder.hpp"
#include <pthread.h>
#include <stdlib.h>
#pragma GCC push_options
#pragma GCC diagnostic ignored "-Wsign-conversion"
#include <bson.h>
#include <mongoc.h>
#pragma GCC pop_options
#include "../raii.hpp"
#include "../log.hpp"
#include "../time.hpp"
#include "../system_exception.hpp"
#include "../uuid.hpp"

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

		struct CollectionCloser {
			CONSTEXPR ::mongoc_collection_t *operator()() const NOEXCEPT {
				return NULLPTR;
			}
			void operator()(::mongoc_collection_t *collection) const NOEXCEPT {
				::mongoc_collection_destroy(collection);
			}
		};
		struct CursorCloser {
			CONSTEXPR ::mongoc_cursor_t *operator()() const NOEXCEPT {
				return NULLPTR;
			}
			void operator()(::mongoc_cursor_t *cursor) const NOEXCEPT {
				::mongoc_cursor_destroy(cursor);
			}
		};

		struct BsonGuard : NONCOPYABLE {
			std::string data;
			::bson_t &bson;

			BsonGuard(::bson_t &raw, const BsonBuilder &builder)
				: data(builder.build(false)), bson(raw)
			{
				if(!::bson_init_static(&bson, reinterpret_cast<const unsigned char *>(data.data()), data.size())){
					DEBUG_THROW(SystemException, EINVAL);
				}
			}
			~BsonGuard(){
				::bson_destroy(&bson);
			}
		};

#define DEBUG_THROW_MONGODB_EXCEPTION(bson_err_, database_)	\
	DEBUG_THROW(::Poseidon::MongoDb::Exception, database_, (bson_err_).code, ::Poseidon::SharedNts((bson_err_).message))

		class DelegatedConnection : public Connection {
		private:
			const SharedNts m_database;

			UniqueHandle<UriCloser> m_uri;
			UniqueHandle<ClientCloser> m_client;

			UniqueHandle<CollectionCloser> m_collection;
			UniqueHandle<CursorCloser> m_cursor;
			const ::bson_t *m_cursor_head;

		public:
			DelegatedConnection(const char *server_addr, unsigned server_port,
				const char *user_name, const char *password, const char *auth_database, bool use_ssl, const char *database)
				: m_database(database)
				, m_cursor_head(NULLPTR)
			{
				const int err = ::pthread_once(&g_mongo_once, &init_mongo);
				if(err != 0){
					LOG_POSEIDON_FATAL("::pthread_once() failed with error code ", err);
					std::abort();
				}

				if(!m_uri.reset(::mongoc_uri_new_for_host_port(server_addr, server_port))){
					DEBUG_THROW(SystemException, ENOMEM);
				}
				if(!::mongoc_uri_set_username(m_uri.get(), user_name)){
					LOG_POSEIDON_ERROR("Failed to set MongoDB user name: ", user_name);
					DEBUG_THROW(SystemException, EINVAL);
				}
				if(!::mongoc_uri_set_password(m_uri.get(), password)){
					LOG_POSEIDON_ERROR("Failed to set MongoDB password: ", password);
					DEBUG_THROW(SystemException, EINVAL);
				}
				if(!::mongoc_uri_set_database(m_uri.get(), auth_database)){
					LOG_POSEIDON_ERROR("Failed to set MongoDB authentication database: ", auth_database);
					DEBUG_THROW(SystemException, EINVAL);
				}
				if(!::mongoc_uri_set_option_as_bool(m_uri.get(), "ssl", use_ssl)){
					DEBUG_THROW(SystemException, EINVAL);
				}

				if(!m_client.reset(::mongoc_client_new_from_uri(m_uri.get()))){
					DEBUG_THROW(SystemException, ENOMEM);
				}
			}

		public:
			void do_execute_command(const char *collection, const BsonBuilder &query, boost::uint32_t begin, boost::uint32_t limit){
				do_discard_result();

				if(!m_collection.reset(::mongoc_client_get_collection(m_client.get(), m_database.get(), collection))){
					DEBUG_THROW(SystemException, ENOMEM);
				}

				::bson_t query_bson;
				const BsonGuard query_guard(query_bson, query);
				boost::uint32_t flags = MONGOC_QUERY_NONE;
				if(!m_cursor.reset(::mongoc_collection_command(m_collection.get(),
					static_cast< ::mongoc_query_flags_t>(flags), begin, limit, 0, &query_bson, NULLPTR, NULLPTR)))
				{
					DEBUG_THROW(SystemException, ENOMEM);
				}
			}
			void do_execute_query(const char *collection, const BsonBuilder &query, boost::uint32_t begin, boost::uint32_t limit){
				do_discard_result();

				if(!m_collection.reset(::mongoc_client_get_collection(m_client.get(), m_database.get(), collection))){
					DEBUG_THROW(SystemException, ENOMEM);
				}

				::bson_t query_bson;
				const BsonGuard query_guard(query_bson, query);
				boost::uint32_t flags = MONGOC_QUERY_NONE;
				if(!m_cursor.reset(::mongoc_collection_find(m_collection.get(),
					static_cast< ::mongoc_query_flags_t>(flags), begin, limit, 0, &query_bson, NULLPTR, NULLPTR)))
				{
					DEBUG_THROW(SystemException, ENOMEM);
				}
			}
			void do_execute_insert(const char *collection, const BsonBuilder &doc, bool continue_on_error){
				do_discard_result();

				if(!m_collection.reset(::mongoc_client_get_collection(m_client.get(), m_database.get(), collection))){
					DEBUG_THROW(SystemException, ENOMEM);
				}

				::bson_error_t error;
				::bson_t doc_bson;
				const BsonGuard doc_guard(doc_bson, doc);
				boost::uint32_t flags = MONGOC_INSERT_NONE;
				if(continue_on_error){
					flags |= MONGOC_INSERT_CONTINUE_ON_ERROR;
				}
				if(!::mongoc_collection_insert(m_collection.get(),
					static_cast< ::mongoc_insert_flags_t>(flags), &doc_bson, NULLPTR, &error))
				{
					DEBUG_THROW_MONGODB_EXCEPTION(error, m_database);
				}
			}
			void do_execute_update(const char *collection, const BsonBuilder &query, const BsonBuilder &doc, bool upsert, bool update_all){
				do_discard_result();

				if(!m_collection.reset(::mongoc_client_get_collection(m_client.get(), m_database.get(), collection))){
					DEBUG_THROW(SystemException, ENOMEM);
				}

				::bson_error_t error;
				::bson_t query_bson;
				const BsonGuard query_guard(query_bson, query);
				::bson_t doc_bson;
				const BsonGuard doc_guard(doc_bson, doc);
				boost::uint32_t flags = MONGOC_UPDATE_NONE;
				if(upsert){
					flags |= MONGOC_UPDATE_UPSERT;
				}
				if(update_all){
					flags |= MONGOC_UPDATE_MULTI_UPDATE;
				}
				if(!::mongoc_collection_update(m_collection.get(),
					static_cast< ::mongoc_update_flags_t>(flags), &query_bson, &doc_bson, NULLPTR, &error))
				{
					DEBUG_THROW_MONGODB_EXCEPTION(error, m_database);
				}
			}
			void do_execute_delete(const char *collection, const BsonBuilder &query, bool delete_all){
				do_discard_result();

				if(!m_collection.reset(::mongoc_client_get_collection(m_client.get(), m_database.get(), collection))){
					DEBUG_THROW(SystemException, ENOMEM);
				}

				::bson_error_t error;
				::bson_t query_bson;
				const BsonGuard query_guard(query_bson, query);
				boost::uint32_t flags = MONGOC_REMOVE_NONE;
				if(!delete_all){
					flags |= MONGOC_REMOVE_SINGLE_REMOVE;
				}
				if(!::mongoc_collection_remove(m_collection.get(),
					static_cast< ::mongoc_remove_flags_t>(flags), &query_bson, NULLPTR, &error))
				{
					DEBUG_THROW_MONGODB_EXCEPTION(error, m_database);
				}
			}
			void do_discard_result() NOEXCEPT {
				m_cursor_head = NULLPTR;
				m_cursor.reset();
				m_collection.reset();
			}

			bool do_fetch_next(){
				if(!m_cursor){
					LOG_POSEIDON_DEBUG("Empty set returned from MongoDB server.");
					return false;
				}
				if(!::mongoc_cursor_next(m_cursor.get(), &m_cursor_head)){
					m_cursor_head = NULLPTR;
					::bson_error_t error;
					if(::mongoc_cursor_error(m_cursor.get(), &error)){
						DEBUG_THROW_MONGODB_EXCEPTION(error, m_database);
					}
					return false;
				}
				return true;
			}

#define CHECK_TYPE(iter_, field_, type_)	\
				{	\
					const AUTO(t_, ::bson_iter_type(&(iter_)));	\
					if(t_ != (type_)){	\
						LOG_POSEIDON_ERROR("BSON type mismatch: field = ", (field_), ", type = ", #type_,	\
							", expecting ", static_cast<int>(type_), ", got ", static_cast<int>(t_));	\
						DEBUG_THROW(BasicException, sslit("BSON type mismatch"));	\
					}	\
				}

			Oid do_get_oid(const char *name) const {
				if(!m_cursor_head){
					DEBUG_THROW(BasicException, sslit("No more results"));
				}
				::bson_iter_t iter;
				if(!::bson_iter_init_find(&iter, m_cursor_head, name)){
					DEBUG_THROW(BasicException, sslit("Field not found"));
				}
				CHECK_TYPE(iter, name, BSON_TYPE_OID)
				return Oid(::bson_iter_oid(&iter)->bytes);
			}
			bool do_get_boolean(const char *name) const {
				if(!m_cursor_head){
					DEBUG_THROW(BasicException, sslit("No more results"));
				}
				::bson_iter_t iter;
				if(!::bson_iter_init_find(&iter, m_cursor_head, name)){
					DEBUG_THROW(BasicException, sslit("Field not found"));
				}
				CHECK_TYPE(iter, name, BSON_TYPE_BOOL)
				return ::bson_iter_bool(&iter);
			}
			boost::int64_t do_get_signed(const char *name) const {
				if(!m_cursor_head){
					DEBUG_THROW(BasicException, sslit("No more results"));
				}
				::bson_iter_t iter;
				if(!::bson_iter_init_find(&iter, m_cursor_head, name)){
					DEBUG_THROW(BasicException, sslit("Field not found"));
				}
				CHECK_TYPE(iter, name, BSON_TYPE_INT64)
				return ::bson_iter_int64(&iter);
			}
			boost::uint64_t do_get_unsigned(const char *name) const {
				if(!m_cursor_head){
					DEBUG_THROW(BasicException, sslit("No more results"));
				}
				::bson_iter_t iter;
				if(!::bson_iter_init_find(&iter, m_cursor_head, name)){
					DEBUG_THROW(BasicException, sslit("Field not found"));
				}
				CHECK_TYPE(iter, name, BSON_TYPE_INT64)
				return static_cast<boost::uint64_t>(::bson_iter_int64(&iter));
			}
			double do_get_double(const char *name) const {
				if(!m_cursor_head){
					DEBUG_THROW(BasicException, sslit("No more results"));
				}
				::bson_iter_t iter;
				if(!::bson_iter_init_find(&iter, m_cursor_head, name)){
					DEBUG_THROW(BasicException, sslit("Field not found"));
				}
				CHECK_TYPE(iter, name, BSON_TYPE_DOUBLE)
				return static_cast<double>(::bson_iter_int64(&iter));
			}
			std::string do_get_string(const char *name) const {
				if(!m_cursor_head){
					DEBUG_THROW(BasicException, sslit("No more results"));
				}
				::bson_iter_t iter;
				if(!::bson_iter_init_find(&iter, m_cursor_head, name)){
					DEBUG_THROW(BasicException, sslit("Field not found"));
				}
				CHECK_TYPE(iter, name, BSON_TYPE_UTF8)
				boost::uint32_t len;
				const char *const str = ::bson_iter_utf8(&iter, &len);
				return std::string(str, len);
			}
			boost::uint64_t do_get_datetime(const char *name) const {
				if(!m_cursor_head){
					DEBUG_THROW(BasicException, sslit("No more results"));
				}
				::bson_iter_t iter;
				if(!::bson_iter_init_find(&iter, m_cursor_head, name)){
					DEBUG_THROW(BasicException, sslit("Field not found"));
				}
				CHECK_TYPE(iter, name, BSON_TYPE_UTF8)
				const AUTO(str, ::bson_iter_utf8(&iter, NULLPTR));
				return scan_time(str);
			}
			Uuid do_get_uuid(const char *name) const {
				if(!m_cursor_head){
					DEBUG_THROW(BasicException, sslit("No more results"));
				}
				::bson_iter_t iter;
				if(!::bson_iter_init_find(&iter, m_cursor_head, name)){
					DEBUG_THROW(BasicException, sslit("Field not found"));
				}
				CHECK_TYPE(iter, name, BSON_TYPE_UTF8)
				boost::uint32_t len;
				const char *const str = ::bson_iter_utf8(&iter, &len);
				if(len != 36){
					DEBUG_THROW(BasicException, sslit("Unexpected UUID string length"));
				}
				return Uuid(reinterpret_cast<const char (&)[36]>(*str));
			}
			std::string do_get_blob(const char *name) const {
				if(!m_cursor_head){
					DEBUG_THROW(BasicException, sslit("No more results"));
				}
				::bson_iter_t iter;
				if(!::bson_iter_init_find(&iter, m_cursor_head, name)){
					DEBUG_THROW(BasicException, sslit("Field not found"));
				}
				CHECK_TYPE(iter, name, BSON_TYPE_BINARY)
				boost::uint32_t len;
				const boost::uint8_t *data;
				::bson_iter_binary(&iter, NULLPTR, &len, &data);
				return std::string(reinterpret_cast<const char *>(data), len);
			}
		};
	}

	boost::shared_ptr<Connection> Connection::create(const char *server_addr, unsigned server_port,
		const char *user_name, const char *password, const char *auth_database, bool use_ssl, const char *database)
	{
		return boost::make_shared<DelegatedConnection>(server_addr, server_port,
			user_name, password, auth_database, use_ssl, database);
	}

	Connection::~Connection(){
	}

	void Connection::execute_command(const char *collection, const BsonBuilder &query, boost::uint32_t begin, boost::uint32_t limit){
		static_cast<DelegatedConnection &>(*this).do_execute_command(collection, query, begin, limit);
	}
	void Connection::execute_query(const char *collection, const BsonBuilder &query, boost::uint32_t begin, boost::uint32_t limit){
		static_cast<DelegatedConnection &>(*this).do_execute_query(collection, query, begin, limit);
	}
	void Connection::execute_insert(const char *collection, const BsonBuilder &doc, bool continue_on_error){
		static_cast<DelegatedConnection &>(*this).do_execute_insert(collection, doc, continue_on_error);
	}
	void Connection::execute_update(const char *collection, const BsonBuilder &query, const BsonBuilder &doc, bool upsert, bool update_all){
		static_cast<DelegatedConnection &>(*this).do_execute_update(collection, query, doc, upsert, update_all);
	}
	void Connection::execute_delete(const char *collection, const BsonBuilder &query, bool delete_all){
		static_cast<DelegatedConnection &>(*this).do_execute_delete(collection, query, delete_all);
	}
	void Connection::discard_result() NOEXCEPT {
		static_cast<DelegatedConnection &>(*this).do_discard_result();
	}

	bool Connection::fetch_next(){
		return static_cast<DelegatedConnection &>(*this).do_fetch_next();
	}

	Oid Connection::get_oid(const char *name) const {
		return static_cast<const DelegatedConnection &>(*this).do_get_oid(name);
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
	std::string Connection::get_blob(const char *name) const {
		return static_cast<const DelegatedConnection &>(*this).do_get_blob(name);
	}
}

}
