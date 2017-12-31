// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#ifndef OBJECT_NAME
#  error OBJECT_NAME is undefined.
#endif

#ifndef OBJECT_FIELDS
#  error OBJECT_FIELDS is undefined.
#endif

#ifndef OBJECT_PRIMARY_KEY
#  error OBJECT_PRIMARY_KEY is undefined.
#endif

#ifndef POSEIDON_MONGODB_OBJECT_BASE_HPP_
#  error Please #include <poseidon/mongodb/object_base.hpp> first.
#endif

class OBJECT_NAME : public ::Poseidon::MongoDb::ObjectBase {
public:
	static ::boost::shared_ptr< ::Poseidon::MongoDb::ObjectBase> create(){
		return ::boost::make_shared<OBJECT_NAME>();
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

#define FIELD_BOOLEAN(id_)                ::Poseidon::MongoDb::ObjectBase::Field<bool> id_;
#define FIELD_SIGNED(id_)                 ::Poseidon::MongoDb::ObjectBase::Field< ::boost::int64_t> id_;
#define FIELD_UNSIGNED(id_)               ::Poseidon::MongoDb::ObjectBase::Field< ::boost::uint64_t> id_;
#define FIELD_DOUBLE(id_)                 ::Poseidon::MongoDb::ObjectBase::Field<double> id_;
#define FIELD_STRING(id_)                 ::Poseidon::MongoDb::ObjectBase::Field< ::std::string> id_;
#define FIELD_DATETIME(id_)               ::Poseidon::MongoDb::ObjectBase::Field< ::boost::uint64_t> id_;
#define FIELD_UUID(id_)                   ::Poseidon::MongoDb::ObjectBase::Field< ::Poseidon::Uuid> id_;
#define FIELD_BLOB(id_)                   ::Poseidon::MongoDb::ObjectBase::Field< ::std::basic_string<unsigned char> > id_;

	OBJECT_FIELDS

public:
	OBJECT_NAME();

#undef FIELD_BOOLEAN
#undef FIELD_SIGNED
#undef FIELD_UNSIGNED
#undef FIELD_DOUBLE
#undef FIELD_STRING
#undef FIELD_DATETIME
#undef FIELD_UUID
#undef FIELD_BLOB

#define FIELD_BOOLEAN(id_)                , bool id_ ## X_
#define FIELD_SIGNED(id_)                 , ::boost::int64_t id_ ## X_
#define FIELD_UNSIGNED(id_)               , ::boost::uint64_t id_ ## X_
#define FIELD_DOUBLE(id_)                 , double id_ ## X_
#define FIELD_STRING(id_)                 , ::std::string id_ ## X_
#define FIELD_DATETIME(id_)               , ::boost::uint64_t id_ ## X_
#define FIELD_UUID(id_)                   , const ::Poseidon::Uuid & id_ ## X_
#define FIELD_BLOB(id_)                   , ::std::basic_string<unsigned char> id_ ## X_

	explicit OBJECT_NAME(STRIP_FIRST(void OBJECT_FIELDS));

	~OBJECT_NAME() OVERRIDE;

public:
	const char *get_collection() const OVERRIDE;
	void generate_document(::Poseidon::MongoDb::BsonBuilder &doc_) const OVERRIDE;
	::std::string generate_primary_key() const OVERRIDE;
	void fetch(const ::boost::shared_ptr<const ::Poseidon::MongoDb::Connection> &conn_) OVERRIDE;
};

#ifdef MONGODB_OBJECT_EMIT_EXTERNAL_DEFINITIONS
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wshadow"

OBJECT_NAME::OBJECT_NAME()
	: ::Poseidon::MongoDb::ObjectBase()

#undef FIELD_BOOLEAN
#undef FIELD_SIGNED
#undef FIELD_UNSIGNED
#undef FIELD_DOUBLE
#undef FIELD_STRING
#undef FIELD_DATETIME
#undef FIELD_UUID
#undef FIELD_BLOB

#define FIELD_BOOLEAN(id_)                , id_(this)
#define FIELD_SIGNED(id_)                 , id_(this)
#define FIELD_UNSIGNED(id_)               , id_(this)
#define FIELD_DOUBLE(id_)                 , id_(this)
#define FIELD_STRING(id_)                 , id_(this)
#define FIELD_DATETIME(id_)               , id_(this)
#define FIELD_UUID(id_)                   , id_(this)
#define FIELD_BLOB(id_)                   , id_(this)

	OBJECT_FIELDS
{ }

#undef FIELD_BOOLEAN
#undef FIELD_SIGNED
#undef FIELD_UNSIGNED
#undef FIELD_DOUBLE
#undef FIELD_STRING
#undef FIELD_DATETIME
#undef FIELD_UUID
#undef FIELD_BLOB

#define FIELD_BOOLEAN(id_)                , bool id_ ## X_
#define FIELD_SIGNED(id_)                 , ::boost::int64_t id_ ## X_
#define FIELD_UNSIGNED(id_)               , ::boost::uint64_t id_ ## X_
#define FIELD_DOUBLE(id_)                 , double id_ ## X_
#define FIELD_STRING(id_)                 , ::std::string id_ ## X_
#define FIELD_DATETIME(id_)               , ::boost::uint64_t id_ ## X_
#define FIELD_UUID(id_)                   , const ::Poseidon::Uuid & id_ ## X_
#define FIELD_BLOB(id_)                   , ::std::basic_string<unsigned char> id_ ## X_

OBJECT_NAME::OBJECT_NAME(STRIP_FIRST(void OBJECT_FIELDS))
	: ::Poseidon::MongoDb::ObjectBase()

#undef FIELD_BOOLEAN
#undef FIELD_SIGNED
#undef FIELD_UNSIGNED
#undef FIELD_DOUBLE
#undef FIELD_STRING
#undef FIELD_DATETIME
#undef FIELD_UUID
#undef FIELD_BLOB

#define FIELD_BOOLEAN(id_)                , id_(this, id_ ## X_)
#define FIELD_SIGNED(id_)                 , id_(this, id_ ## X_)
#define FIELD_UNSIGNED(id_)               , id_(this, id_ ## X_)
#define FIELD_DOUBLE(id_)                 , id_(this, id_ ## X_)
#define FIELD_STRING(id_)                 , id_(this, STD_MOVE(id_ ## X_))
#define FIELD_DATETIME(id_)               , id_(this, id_ ## X_)
#define FIELD_UUID(id_)                   , id_(this, id_ ## X_)
#define FIELD_BLOB(id_)                   , id_(this, STD_MOVE(id_ ## X_))

	OBJECT_FIELDS
{
	::Poseidon::atomic_fence(::Poseidon::ATOMIC_RELEASE);
}

OBJECT_NAME::~OBJECT_NAME(){ }

const char *OBJECT_NAME::get_collection() const OVERRIDE {
	return TOKEN_TO_STR(OBJECT_NAME);
}
void OBJECT_NAME::generate_document(::Poseidon::MongoDb::BsonBuilder &doc_) const {
	PROFILE_ME;

	const ::Poseidon::RecursiveMutex::UniqueLock lock_(m_mutex);

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

#define FIELD_BOOLEAN(id_)                doc_.append_boolean  (::Poseidon::SharedNts::view(TOKEN_TO_STR(id_)), id_);
#define FIELD_SIGNED(id_)                 doc_.append_signed   (::Poseidon::SharedNts::view(TOKEN_TO_STR(id_)), id_);
#define FIELD_UNSIGNED(id_)               doc_.append_unsigned (::Poseidon::SharedNts::view(TOKEN_TO_STR(id_)), id_);
#define FIELD_DOUBLE(id_)                 doc_.append_double   (::Poseidon::SharedNts::view(TOKEN_TO_STR(id_)), id_);
#define FIELD_STRING(id_)                 doc_.append_string   (::Poseidon::SharedNts::view(TOKEN_TO_STR(id_)), id_);
#define FIELD_DATETIME(id_)               doc_.append_datetime (::Poseidon::SharedNts::view(TOKEN_TO_STR(id_)), id_);
#define FIELD_UUID(id_)                   doc_.append_uuid     (::Poseidon::SharedNts::view(TOKEN_TO_STR(id_)), id_);
#define FIELD_BLOB(id_)                   doc_.append_blob     (::Poseidon::SharedNts::view(TOKEN_TO_STR(id_)), id_);

	OBJECT_FIELDS
}
::std::string OBJECT_NAME::generate_primary_key() const {
	PROFILE_ME;

	const ::Poseidon::RecursiveMutex::UniqueLock lock_(m_mutex);

	OBJECT_PRIMARY_KEY
}
void fetch(const ::boost::shared_ptr<const ::Poseidon::MongoDb::Connection> &conn_){
	PROFILE_ME;

	const ::Poseidon::RecursiveMutex::UniqueLock lock_(m_mutex);

#undef FIELD_BOOLEAN
#undef FIELD_SIGNED
#undef FIELD_UNSIGNED
#undef FIELD_DOUBLE
#undef FIELD_STRING
#undef FIELD_DATETIME
#undef FIELD_UUID
#undef FIELD_BLOB

#define FIELD_BOOLEAN(id_)                id_.set(conn_->get_boolean  ( TOKEN_TO_STR(id_) ), false);
#define FIELD_SIGNED(id_)                 id_.set(conn_->get_signed   ( TOKEN_TO_STR(id_) ), false);
#define FIELD_UNSIGNED(id_)               id_.set(conn_->get_unsigned ( TOKEN_TO_STR(id_) ), false);
#define FIELD_DOUBLE(id_)                 id_.set(conn_->get_double   ( TOKEN_TO_STR(id_) ), false);
#define FIELD_STRING(id_)                 id_.set(conn_->get_string   ( TOKEN_TO_STR(id_) ), false);
#define FIELD_DATETIME(id_)               id_.set(conn_->get_datetime ( TOKEN_TO_STR(id_) ), false);
#define FIELD_UUID(id_)                   id_.set(conn_->get_uuid     ( TOKEN_TO_STR(id_) ), false);
#define FIELD_BLOB(id_)                   id_.set(conn_->get_blob     ( TOKEN_TO_STR(id_) ), false);

	OBJECT_FIELDS
}

#pragma GCC diagnostic pop
#endif // MONGODB_OBJECT_EMIT_EXTERNAL_DEFINITIONS

#undef OBJECT_NAME
#undef OBJECT_FIELDS
#undef OBJECT_PRIMARY_KEY
