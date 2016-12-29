// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2016, LH_Mouse. All wrongs reserved.

#ifndef MONGODB_OBJECT_NAME
#   error MONGODB_OBJECT_NAME is undefined.
#endif

#ifndef MONGODB_OBJECT_FIELDS
#   error MONGODB_OBJECT_FIELDS is undefined.
#endif

#ifndef MONGODB_OBJECT_PRIMARY_KEY
#   error MONGODB_OBJECT_PRIMARY_KEY is undefined.
#endif

#ifndef POSEIDON_MONGODB_OBJECT_BASE_HPP_
#   error Please #include <poseidon/mongodb/object_base.hpp> first.
#endif

#pragma GCC push_options
#pragma GCC diagnostic ignored "-Wshadow"

class MONGODB_OBJECT_NAME : public ::Poseidon::MongoDb::ObjectBase {
public:
	static ::boost::shared_ptr< ::Poseidon::MongoDb::ObjectBase> create(){
		return ::boost::make_shared<MONGODB_OBJECT_NAME>();
	}

public:

#undef FIELD_BOOLEAN
#undef FIELD_SIGNED
#undef FIELD_UNSIGNED
#undef FIELD_DOUBLE
#undef FIELD_STRING
#undef FIELD_DATETIME
#undef FIELD_UUID
#undef FIELD_BLOB

#define FIELD_BOOLEAN(name_)                ::Poseidon::MongoDb::ObjectBase::Field<bool> name_;
#define FIELD_SIGNED(name_)                 ::Poseidon::MongoDb::ObjectBase::Field< ::boost::int64_t> name_;
#define FIELD_UNSIGNED(name_)               ::Poseidon::MongoDb::ObjectBase::Field< ::boost::uint64_t> name_;
#define FIELD_DOUBLE(name_)                 ::Poseidon::MongoDb::ObjectBase::Field<double> name_;
#define FIELD_STRING(name_)                 ::Poseidon::MongoDb::ObjectBase::Field< ::std::string> name_;
#define FIELD_DATETIME(name_)               ::Poseidon::MongoDb::ObjectBase::Field< ::boost::uint64_t> name_;
#define FIELD_UUID(name_)                   ::Poseidon::MongoDb::ObjectBase::Field< ::Poseidon::Uuid> name_;
#define FIELD_BLOB(name_)                   ::Poseidon::MongoDb::ObjectBase::Field< ::std::string> name_;

	MONGODB_OBJECT_FIELDS

public:
	MONGODB_OBJECT_NAME()
		: ::Poseidon::MongoDb::ObjectBase()

#undef FIELD_BOOLEAN
#undef FIELD_SIGNED
#undef FIELD_UNSIGNED
#undef FIELD_DOUBLE
#undef FIELD_STRING
#undef FIELD_DATETIME
#undef FIELD_UUID
#undef FIELD_BLOB

#define FIELD_BOOLEAN(name_)                , name_(this)
#define FIELD_SIGNED(name_)                 , name_(this)
#define FIELD_UNSIGNED(name_)               , name_(this)
#define FIELD_DOUBLE(name_)                 , name_(this)
#define FIELD_STRING(name_)                 , name_(this)
#define FIELD_DATETIME(name_)               , name_(this)
#define FIELD_UUID(name_)                   , name_(this)
#define FIELD_BLOB(name_)                   , name_(this)

		MONGODB_OBJECT_FIELDS
	{
	}

#undef FIELD_BOOLEAN
#undef FIELD_SIGNED
#undef FIELD_UNSIGNED
#undef FIELD_DOUBLE
#undef FIELD_STRING
#undef FIELD_DATETIME
#undef FIELD_UUID
#undef FIELD_BLOB

#define FIELD_BOOLEAN(name_)                , bool name_ ## X_
#define FIELD_SIGNED(name_)                 , ::boost::int64_t name_ ## X_
#define FIELD_UNSIGNED(name_)               , ::boost::uint64_t name_ ## X_
#define FIELD_DOUBLE(name_)                 , double name_ ## X_
#define FIELD_STRING(name_)                 , ::std::string name_ ## X_
#define FIELD_DATETIME(name_)               , ::boost::uint64_t name_ ## X_
#define FIELD_UUID(name_)                   , const ::Poseidon::Uuid & name_ ## X_
#define FIELD_BLOB(name_)                   , ::std::string name_ ## X_

	explicit MONGODB_OBJECT_NAME(STRIP_FIRST(void MONGODB_OBJECT_FIELDS))
		: ::Poseidon::MongoDb::ObjectBase()

#undef FIELD_BOOLEAN
#undef FIELD_SIGNED
#undef FIELD_UNSIGNED
#undef FIELD_DOUBLE
#undef FIELD_STRING
#undef FIELD_DATETIME
#undef FIELD_UUID
#undef FIELD_BLOB

#define FIELD_BOOLEAN(name_)                , name_(this, name_ ## X_)
#define FIELD_SIGNED(name_)                 , name_(this, name_ ## X_)
#define FIELD_UNSIGNED(name_)               , name_(this, name_ ## X_)
#define FIELD_DOUBLE(name_)                 , name_(this, name_ ## X_)
#define FIELD_STRING(name_)                 , name_(this, STD_MOVE(name_ ## X_))
#define FIELD_DATETIME(name_)               , name_(this, name_ ## X_)
#define FIELD_UUID(name_)                   , name_(this, name_ ## X_)
#define FIELD_BLOB(name_)                   , name_(this, STD_MOVE(name_ ## X_))

		MONGODB_OBJECT_FIELDS
	{
		::Poseidon::atomic_fence(::Poseidon::ATOMIC_RELEASE);
	}

public:
	const char *get_collection_name() const OVERRIDE {
		return TOKEN_TO_STR(MONGODB_OBJECT_NAME);
	}

	void generate_document(::Poseidon::MongoDb::BsonBuilder &doc_) const OVERRIDE {
		AUTO(pkey_, generate_primary_key());
		if(!pkey_.empty()){
			doc_.append_string(::Poseidon::SharedNts::view("_id"), STD_MOVE(pkey_));
		}

#undef FIELD_BOOLEAN
#undef FIELD_SIGNED
#undef FIELD_UNSIGNED
#undef FIELD_DOUBLE
#undef FIELD_STRING
#undef FIELD_DATETIME
#undef FIELD_UUID
#undef FIELD_BLOB

#define FIELD_BOOLEAN(name_)                doc_.append_boolean  (::Poseidon::SharedNts::view(TOKEN_TO_STR(name_)), name_);
#define FIELD_SIGNED(name_)                 doc_.append_signed   (::Poseidon::SharedNts::view(TOKEN_TO_STR(name_)), name_);
#define FIELD_UNSIGNED(name_)               doc_.append_unsigned (::Poseidon::SharedNts::view(TOKEN_TO_STR(name_)), name_);
#define FIELD_DOUBLE(name_)                 doc_.append_double   (::Poseidon::SharedNts::view(TOKEN_TO_STR(name_)), name_);
#define FIELD_STRING(name_)                 doc_.append_string   (::Poseidon::SharedNts::view(TOKEN_TO_STR(name_)), name_);
#define FIELD_DATETIME(name_)               doc_.append_datetime (::Poseidon::SharedNts::view(TOKEN_TO_STR(name_)), name_);
#define FIELD_UUID(name_)                   doc_.append_uuid     (::Poseidon::SharedNts::view(TOKEN_TO_STR(name_)), name_);
#define FIELD_BLOB(name_)                   doc_.append_blob     (::Poseidon::SharedNts::view(TOKEN_TO_STR(name_)), name_);

		MONGODB_OBJECT_FIELDS
	}
	::std::string generate_primary_key() const OVERRIDE {
		const ::Poseidon::RecursiveMutex::UniqueLock lock_(m_mutex);

		MONGODB_OBJECT_PRIMARY_KEY
	}
	void fetch(const ::boost::shared_ptr<const ::Poseidon::MongoDb::Connection> &conn_) OVERRIDE {

#undef FIELD_BOOLEAN
#undef FIELD_SIGNED
#undef FIELD_UNSIGNED
#undef FIELD_DOUBLE
#undef FIELD_STRING
#undef FIELD_DATETIME
#undef FIELD_UUID
#undef FIELD_BLOB

#define FIELD_BOOLEAN(name_)                name_.set(conn_->get_boolean  ( TOKEN_TO_STR(name_) ), false);
#define FIELD_SIGNED(name_)                 name_.set(conn_->get_signed   ( TOKEN_TO_STR(name_) ), false);
#define FIELD_UNSIGNED(name_)               name_.set(conn_->get_unsigned ( TOKEN_TO_STR(name_) ), false);
#define FIELD_DOUBLE(name_)                 name_.set(conn_->get_double   ( TOKEN_TO_STR(name_) ), false);
#define FIELD_STRING(name_)                 name_.set(conn_->get_string   ( TOKEN_TO_STR(name_) ), false);
#define FIELD_DATETIME(name_)               name_.set(conn_->get_datetime ( TOKEN_TO_STR(name_) ), false);
#define FIELD_UUID(name_)                   name_.set(conn_->get_uuid     ( TOKEN_TO_STR(name_) ), false);
#define FIELD_BLOB(name_)                   name_.set(conn_->get_blob     ( TOKEN_TO_STR(name_) ), false);

		MONGODB_OBJECT_FIELDS
	}
};

#pragma GCC pop_options

#undef MONGODB_OBJECT_NAME
#undef MONGODB_OBJECT_FIELDS
#undef MONGODB_OBJECT_PRIMARY_KEY
