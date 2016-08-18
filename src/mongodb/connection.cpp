// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2016, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "connection.hpp"
#include "exception.hpp"
#include "oid.hpp"
#include "bson_builder.hpp"
#include "../raii.hpp"
#include "../log.hpp"
#include "../time.hpp"
#include "../system_exception.hpp"
#include "../uuid.hpp"
#include <mongo-client/bson.h>
#include <mongo-client/mongo.h>

namespace Poseidon {

namespace MongoDb {
	namespace {
		struct SyncConnectionCloser {
			CONSTEXPR ::mongo_sync_connection *operator()() const NOEXCEPT {
				return NULLPTR;
			}
			void operator()(::mongo_sync_connection *conn) const NOEXCEPT {
				::mongo_sync_disconnect(conn);
			}
		};

		struct SyncCursorDeleter {
			CONSTEXPR ::mongo_sync_cursor *operator()() const NOEXCEPT {
				return NULLPTR;
			}
			void operator()(::mongo_sync_cursor *cursor) const NOEXCEPT {
				::mongo_sync_cursor_free(cursor);
			}
		};

		struct BsonDeleter {
			CONSTEXPR ::bson *operator()() const NOEXCEPT {
				return NULLPTR;
			}
			void operator()(::bson *b) const NOEXCEPT {
				::bson_free(b);
			}
		};

		struct BsonCursorDeleter {
			CONSTEXPR ::bson_cursor *operator()() const NOEXCEPT {
				return NULLPTR;
			}
			void operator()(::bson_cursor *cursor) const NOEXCEPT {
				::bson_cursor_free(cursor);
			}
		};

#define DEBUG_THROW_MONGODB_EXCEPTION_USING_ERRNO(conn_, database_)	\
	do {	\
		const int err_ = errno;	\
		::Poseidon::SharedNts shared_msg_;	\
		char *msg_;	\
		if(!::mongo_sync_cmd_get_last_error(conn_, database_, &msg_)){	\
			msg_ = NULLPTR;	\
		}	\
		if(msg_){	\
			shared_msg_ = ::Poseidon::SharedNts(msg_);	\
		} else {	\
			shared_msg_ = ::Poseidon::sslit("<Last Error Not Set>");	\
		}	\
		DEBUG_THROW(::Poseidon::MongoDb::Exception, database_, err_, shared_msg_);	\
	} while(0)

		class DelegatedConnection : public Connection {
		private:
			const SharedNts m_database;

			UniqueHandle<SyncConnectionCloser> m_conn;

			UniqueHandle<SyncCursorDeleter> m_cursor;
			UniqueHandle<BsonDeleter> m_result;
			UniqueHandle<BsonCursorDeleter> m_result_cursor;

		public:
			DelegatedConnection(const char *server_addr, unsigned server_port,
				bool slave_ok, const char *database)
				: m_database(database)
			{
				if(!m_conn.reset(::mongo_sync_connect(server_addr, static_cast<int>(server_port), slave_ok))){
					DEBUG_THROW(SystemException, errno);
				}

				if(!::mongo_sync_conn_set_auto_reconnect(m_conn.get(), true)){
					DEBUG_THROW(SystemException, errno);
				}
			}

		private:
			std::string format_namspace(const char *collection) const {
				std::string ret;
				ret.reserve(255);
				ret.append(m_database.get());
				ret.push_back('.');
				ret.append(collection);
				return ret;
			}
			UniqueHandle<BsonDeleter> format_bson(const BsonBuilder &builder){
				std::string data = builder.build();
				if(data.size() > 0x7FFFFFFF){
					DEBUG_THROW(ProtocolException,
						sslit("Failed to create BSON object: BSON size is too large"), -1);
				}
				UniqueHandle<BsonDeleter> b;
				if(!b.reset(::bson_new_from_data(reinterpret_cast<const ::guint8 *>(data.data()),
					static_cast< ::gint32>(std::max<std::size_t>(data.size(), 1) - 1))))
				{
					DEBUG_THROW(ProtocolException,
						sslit("Failed to create BSON object: bson_new() failed"), -1);
				}
				if(!::bson_finish(b.get())){
					DEBUG_THROW(ProtocolException,
						sslit("Failed to finish BSON object: bson_finish() failed"), -1);
				}
				return STD_MOVE_IDN(b);
			}

		public:
			void do_execute_insert(const char *collection, const BsonBuilder &data){
				do_discard_result();

				const AUTO(ns, format_namspace(collection));
				const AUTO(bd, format_bson(data));
				if(!::mongo_sync_cmd_insert(m_conn.get(), ns.c_str(), bd.get(), static_cast<void *>(0))){
					DEBUG_THROW_MONGODB_EXCEPTION_USING_ERRNO(m_conn.get(), m_database);
				}
			}
			void do_execute_update(const char *collection, const BsonBuilder &query, bool upsert, bool update_all, const BsonBuilder &data){
				do_discard_result();

				const AUTO(ns, format_namspace(collection));
				const AUTO(bq, format_bson(query));
				 ::gint32 flags = 0;
				if(upsert){
					flags |= MONGO_WIRE_FLAG_UPDATE_UPSERT;
				}
				if(update_all){
					flags |= MONGO_WIRE_FLAG_UPDATE_MULTI;
				}
				const AUTO(bd, format_bson(data));
				if(!::mongo_sync_cmd_update(m_conn.get(), ns.c_str(), flags, bq.get(), bd.get())){
					DEBUG_THROW_MONGODB_EXCEPTION_USING_ERRNO(m_conn.get(), m_database);
				}
			}
			void do_execute_delete(const char *collection, const BsonBuilder &query, bool delete_all){
				do_discard_result();

				const AUTO(ns, format_namspace(collection));
				const AUTO(bq, format_bson(query));
				 ::gint32 flags = 0;
				if(!delete_all){
					flags |= MONGO_WIRE_FLAG_DELETE_SINGLE;
				}
				if(!::mongo_sync_cmd_delete(m_conn.get(), ns.c_str(), flags, bq.get())){
					DEBUG_THROW_MONGODB_EXCEPTION_USING_ERRNO(m_conn.get(), m_database);
				}
			}
			void do_execute_query(const char *collection, const BsonBuilder &query, std::size_t begin, std::size_t limit){
				do_discard_result();

				if(begin > 0x7FFFFFFF){
					DEBUG_THROW(BasicException, sslit("MongoDB: Number of documents to skip is too large"));
				}
				if(limit > 0x7FFFFFFF){
					DEBUG_THROW(BasicException, sslit("MongoDB: Number of documents to return is too large"));
				}

				const AUTO(ns, format_namspace(collection));
				 ::gint32 flags = MONGO_WIRE_FLAG_QUERY_SLAVE_OK;
				const AUTO(bq, format_bson(query));
				AUTO(packet, ::mongo_sync_cmd_query(m_conn.get(), ns.c_str(), flags,
					static_cast< ::gint32>(begin), static_cast< ::gint32>(limit), bq.get(), NULLPTR));
				if(!packet){
					DEBUG_THROW_MONGODB_EXCEPTION_USING_ERRNO(m_conn.get(), m_database);
				}
				if(!m_cursor.reset(::mongo_sync_cursor_new(m_conn.get(), ns.c_str(), packet))){
					LOG_POSEIDON_ERROR("Failed to allocate sync cursor!");
					DEBUG_THROW(BasicException, sslit("MongoDB: Failed to allocate sync cursor"));
				}
			}
			void do_discard_result() NOEXCEPT {
				m_cursor.reset();
				m_result.reset();
				m_result_cursor.reset();
			}

			bool do_fetch_next(){
				if(!m_cursor){
					LOG_POSEIDON_WARNING("No query result cursor?");
					return false;
				}
				if(!::mongo_sync_cursor_next(m_cursor.get())){
					LOG_POSEIDON_DEBUG("No more data available.");
					return false;
				}
				if(!m_result.reset(::mongo_sync_cursor_get_data(m_cursor.get()))){
					LOG_POSEIDON_ERROR("Failed to allocate result BSON object!");
					DEBUG_THROW(BasicException, sslit("MongoDB: Failed to allocate result BSON object"));
				}
				if(!m_result_cursor.reset(::bson_cursor_new(m_result.get()))){
					LOG_POSEIDON_ERROR("Failed to allocate result BSON cursor!");
					DEBUG_THROW(BasicException, sslit("MongoDB: Failed to allocate result BSON cursor"));
				}
				if(!::bson_cursor_next(m_result_cursor.get())){
					LOG_POSEIDON_ERROR("Failed to reset result BSON cursor!");
					DEBUG_THROW(BasicException, sslit("MongoDB: Failed to reset result BSON cursor"));
				}
				return true;
			}

			Oid do_get_oid() const {
				if(!m_result_cursor){
					LOG_POSEIDON_WARNING("No result returned from MongoDB server.");
					DEBUG_THROW(BasicException, sslit("MongoDB: No result returned from MongoDB server"));
				}
				if(!::bson_cursor_find(m_result_cursor.get(), "_id")){
					LOG_POSEIDON_ERROR("Field _id not found");
					DEBUG_THROW(BasicException, sslit("Field _id not found"));
				}
				const ::guint8 *val;
				if(!::bson_cursor_get_oid(m_result_cursor.get(), &val)){
					LOG_POSEIDON_ERROR("Unexpected BSON type: type = ", static_cast<int>(::bson_cursor_type(m_result_cursor.get())));
					DEBUG_THROW(BasicException, sslit("Unexpected BSON type"));
				}
				return Oid(reinterpret_cast<const unsigned char (&)[12]>(val[0]));
			}
			bool do_get_boolean(const char *name) const {
				if(!m_result_cursor){
					LOG_POSEIDON_WARNING("No result returned from MongoDB server.");
					DEBUG_THROW(BasicException, sslit("MongoDB: No result returned from MongoDB server"));
				}
				if(!::bson_cursor_find(m_result_cursor.get(), name)){
					LOG_POSEIDON_ERROR("Name not found: ", name);
					DEBUG_THROW(BasicException, sslit("Name not found"));
				}
				::gboolean val;
				if(!::bson_cursor_get_boolean(m_result_cursor.get(), &val)){
					LOG_POSEIDON_ERROR("Unexpected BSON type: type = ", static_cast<int>(::bson_cursor_type(m_result_cursor.get())));
					DEBUG_THROW(BasicException, sslit("Unexpected BSON type"));
				}
				return val;
			}
			boost::int64_t do_get_signed(const char *name) const {
				if(!m_result_cursor){
					LOG_POSEIDON_WARNING("No result returned from MongoDB server.");
					DEBUG_THROW(BasicException, sslit("MongoDB: No result returned from MongoDB server"));
				}
				if(!::bson_cursor_find(m_result_cursor.get(), name)){
					LOG_POSEIDON_ERROR("Name not found: ", name);
					DEBUG_THROW(BasicException, sslit("Name not found"));
				}
				::gint64 val;
				if(!::bson_cursor_get_int64(m_result_cursor.get(), &val)){
					LOG_POSEIDON_ERROR("Unexpected BSON type: type = ", static_cast<int>(::bson_cursor_type(m_result_cursor.get())));
					DEBUG_THROW(BasicException, sslit("Unexpected BSON type"));
				}
				return val;
			}
			boost::uint64_t do_get_unsigned(const char *name) const {
				if(!m_result_cursor){
					LOG_POSEIDON_WARNING("No result returned from MongoDB server.");
					DEBUG_THROW(BasicException, sslit("MongoDB: No result returned from MongoDB server"));
				}
				if(!::bson_cursor_find(m_result_cursor.get(), name)){
					LOG_POSEIDON_ERROR("Name not found: ", name);
					DEBUG_THROW(BasicException, sslit("Name not found"));
				}
				::gint64 val;
				if(!::bson_cursor_get_int64(m_result_cursor.get(), &val)){
					LOG_POSEIDON_ERROR("Unexpected BSON type: type = ", static_cast<int>(::bson_cursor_type(m_result_cursor.get())));
					DEBUG_THROW(BasicException, sslit("Unexpected BSON type"));
				}
				return static_cast<boost::uint64_t>(val);
			}
			double do_get_double(const char *name) const {
				if(!m_result_cursor){
					LOG_POSEIDON_WARNING("No result returned from MongoDB server.");
					DEBUG_THROW(BasicException, sslit("MongoDB: No result returned from MongoDB server"));
				}
				if(!::bson_cursor_find(m_result_cursor.get(), name)){
					LOG_POSEIDON_ERROR("Name not found: ", name);
					DEBUG_THROW(BasicException, sslit("Name not found"));
				}
				::gdouble val;
				if(!::bson_cursor_get_double(m_result_cursor.get(), &val)){
					LOG_POSEIDON_ERROR("Unexpected BSON type: type = ", static_cast<int>(::bson_cursor_type(m_result_cursor.get())));
					DEBUG_THROW(BasicException, sslit("Unexpected BSON type"));
				}
				return val;
			}
			std::string do_get_string(const char *name) const {
				if(!m_result_cursor){
					LOG_POSEIDON_WARNING("No result returned from MongoDB server.");
					DEBUG_THROW(BasicException, sslit("MongoDB: No result returned from MongoDB server"));
				}
				if(!::bson_cursor_find(m_result_cursor.get(), name)){
					LOG_POSEIDON_ERROR("Name not found: ", name);
					DEBUG_THROW(BasicException, sslit("Name not found"));
				}
				const ::gchar *val;
				if(!::bson_cursor_get_string(m_result_cursor.get(), &val)){
					LOG_POSEIDON_ERROR("Unexpected BSON type: type = ", static_cast<int>(::bson_cursor_type(m_result_cursor.get())));
					DEBUG_THROW(BasicException, sslit("Unexpected BSON type"));
				}
				return std::string(val);
			}
			boost::uint64_t do_get_datetime(const char *name) const {
				if(!m_result_cursor){
					LOG_POSEIDON_WARNING("No result returned from MongoDB server.");
					DEBUG_THROW(BasicException, sslit("MongoDB: No result returned from MongoDB server"));
				}
				if(!::bson_cursor_find(m_result_cursor.get(), name)){
					LOG_POSEIDON_ERROR("Name not found: ", name);
					DEBUG_THROW(BasicException, sslit("Name not found"));
				}
				::gint64 val;
				if(!::bson_cursor_get_utc_datetime(m_result_cursor.get(), &val)){
					LOG_POSEIDON_ERROR("Unexpected BSON type: type = ", static_cast<int>(::bson_cursor_type(m_result_cursor.get())));
					DEBUG_THROW(BasicException, sslit("Unexpected BSON type"));
				}
				return static_cast<boost::uint64_t>(val);
			}
			Uuid do_get_uuid(const char *name) const {
				if(!m_result_cursor){
					LOG_POSEIDON_WARNING("No result returned from MongoDB server.");
					DEBUG_THROW(BasicException, sslit("MongoDB: No result returned from MongoDB server"));
				}
				if(!::bson_cursor_find(m_result_cursor.get(), name)){
					LOG_POSEIDON_ERROR("Name not found: ", name);
					DEBUG_THROW(BasicException, sslit("Name not found"));
				}
				const ::gchar *val;
				if(!::bson_cursor_get_string(m_result_cursor.get(), &val)){
					LOG_POSEIDON_ERROR("Unexpected BSON type: type = ", static_cast<int>(::bson_cursor_type(m_result_cursor.get())));
					DEBUG_THROW(BasicException, sslit("Unexpected BSON type"));
				}
				if(std::strlen(val) != 36){
					LOG_POSEIDON_ERROR("Invalid UUID string: ", val);
					DEBUG_THROW(BasicException, sslit("Invalid UUID string"));
				}
				return Uuid(reinterpret_cast<const char (&)[36]>(val[0]));
			}
			std::string do_get_blob(const char *name) const {
				if(!m_result_cursor){
					LOG_POSEIDON_WARNING("No result returned from MongoDB server.");
					DEBUG_THROW(BasicException, sslit("MongoDB: No result returned from MongoDB server"));
				}
				if(!::bson_cursor_find(m_result_cursor.get(), name)){
					LOG_POSEIDON_ERROR("Name not found: ", name);
					DEBUG_THROW(BasicException, sslit("Name not found"));
				}
				::bson_binary_subtype subtype;
				const ::guint8 *data;
				::gint32 size;
				if(!::bson_cursor_get_binary(m_result_cursor.get(), &subtype, &data, &size)){
					LOG_POSEIDON_ERROR("Unexpected BSON type: type = ", static_cast<int>(::bson_cursor_type(m_result_cursor.get())));
					DEBUG_THROW(BasicException, sslit("Unexpected BSON type"));
				}
				return std::string(reinterpret_cast<const char *>(data), static_cast<std::size_t>(size));
			}
			std::string do_get_regex(const char *name) const {
				if(!m_result_cursor){
					LOG_POSEIDON_WARNING("No result returned from MongoDB server.");
					DEBUG_THROW(BasicException, sslit("MongoDB: No result returned from MongoDB server"));
				}
				if(!::bson_cursor_find(m_result_cursor.get(), name)){
					LOG_POSEIDON_ERROR("Name not found: ", name);
					DEBUG_THROW(BasicException, sslit("Name not found"));
				}
				const ::gchar *val, *options;
				if(!::bson_cursor_get_regex(m_result_cursor.get(), &val, &options)){
					LOG_POSEIDON_ERROR("Unexpected BSON type: type = ", static_cast<int>(::bson_cursor_type(m_result_cursor.get())));
					DEBUG_THROW(BasicException, sslit("Unexpected BSON type"));
				}
				return std::string(val);
			}
		};
	}

	boost::shared_ptr<Connection> Connection::create(const char *server_addr, unsigned server_port,
		bool slave_ok, const char *database)
	{
		return boost::make_shared<DelegatedConnection>(server_addr, server_port, slave_ok, database);
	}

	Connection::~Connection(){
	}

	void Connection::execute_insert(const char *collection, const BsonBuilder &data){
		static_cast<DelegatedConnection &>(*this).do_execute_insert(collection, data);
	}
	void Connection::execute_update(const char *collection, const BsonBuilder &query, bool upsert, bool update_all, const BsonBuilder &data){
		static_cast<DelegatedConnection &>(*this).do_execute_update(collection, query, upsert, update_all, data);
	}
	void Connection::execute_delete(const char *collection, const BsonBuilder &query, bool delete_all){
		static_cast<DelegatedConnection &>(*this).do_execute_delete(collection, query, delete_all);
	}
	void Connection::execute_query(const char *collection, const BsonBuilder &query, std::size_t begin, std::size_t limit){
		static_cast<DelegatedConnection &>(*this).do_execute_query(collection, query, begin, limit);
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
	Oid Connection::get_oid() const {
		return static_cast<const DelegatedConnection &>(*this).do_get_oid();
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
	std::string Connection::get_regex(const char *name) const {
		return static_cast<const DelegatedConnection &>(*this).do_get_regex(name);
	}
}

}
