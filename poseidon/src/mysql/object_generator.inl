// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#ifndef OBJECT_NAME
#  error OBJECT_NAME is undefined.
#endif

#ifndef OBJECT_TABLE
#  error OBJECT_TABLE is undefined.
#endif

#ifndef OBJECT_FIELDS
#  error OBJECT_FIELDS is undefined.
#endif

#ifndef POSEIDON_MYSQL_OBJECT_BASE_HPP_
#  error Please #include <poseidon/mysql/object_base.hpp> first.
#endif

class OBJECT_NAME : public ::Poseidon::Mysql::Object_base {
public:

#undef FIELD_BOOLEAN
#undef FIELD_SIGNED
#undef FIELD_UNSIGNED
#undef FIELD_DOUBLE
#undef FIELD_STRING
#undef FIELD_DATETIME
#undef FIELD_UUID
#undef FIELD_BLOB

#define FIELD_BOOLEAN(id_)                ::Poseidon::Mysql::Object_base::Field<bool> id_;
#define FIELD_SIGNED(id_)                 ::Poseidon::Mysql::Object_base::Field< ::std::int64_t> id_;
#define FIELD_UNSIGNED(id_)               ::Poseidon::Mysql::Object_base::Field< ::std::uint64_t> id_;
#define FIELD_DOUBLE(id_)                 ::Poseidon::Mysql::Object_base::Field<double> id_;
#define FIELD_STRING(id_)                 ::Poseidon::Mysql::Object_base::Field< ::std::string> id_;
#define FIELD_DATETIME(id_)               ::Poseidon::Mysql::Object_base::Field< ::std::uint64_t> id_;
#define FIELD_UUID(id_)                   ::Poseidon::Mysql::Object_base::Field< ::Poseidon::Uuid> id_;
#define FIELD_BLOB(id_)                   ::Poseidon::Mysql::Object_base::Field< ::std::basic_string<unsigned char> > id_;

	OBJECT_FIELDS

public:
	OBJECT_NAME();
	~OBJECT_NAME() OVERRIDE;

#undef FIELD_BOOLEAN
#undef FIELD_SIGNED
#undef FIELD_UNSIGNED
#undef FIELD_DOUBLE
#undef FIELD_STRING
#undef FIELD_DATETIME
#undef FIELD_UUID
#undef FIELD_BLOB

#define FIELD_BOOLEAN(id_)                + 1
#define FIELD_SIGNED(id_)                 + 1
#define FIELD_UNSIGNED(id_)               + 1
#define FIELD_DOUBLE(id_)                 + 1
#define FIELD_STRING(id_)                 + 1
#define FIELD_DATETIME(id_)               + 1
#define FIELD_UUID(id_)                   + 1
#define FIELD_BLOB(id_)                   + 1

#if (0 OBJECT_FIELDS)

#undef FIELD_BOOLEAN
#undef FIELD_SIGNED
#undef FIELD_UNSIGNED
#undef FIELD_DOUBLE
#undef FIELD_STRING
#undef FIELD_DATETIME
#undef FIELD_UUID
#undef FIELD_BLOB

#define FIELD_BOOLEAN(id_)                , bool param_##id_##_Zky_
#define FIELD_SIGNED(id_)                 , ::std::int64_t param_##id_##_Zky_
#define FIELD_UNSIGNED(id_)               , ::std::uint64_t param_##id_##_Zky_
#define FIELD_DOUBLE(id_)                 , double param_##id_##_Zky_
#define FIELD_STRING(id_)                 , ::std::string param_##id_##_Zky_
#define FIELD_DATETIME(id_)               , ::std::uint64_t param_##id_##_Zky_
#define FIELD_UUID(id_)                   , const ::Poseidon::Uuid & param_##id_##_Zky_
#define FIELD_BLOB(id_)                   , ::std::basic_string<unsigned char> param_##id_##_Zky_

	explicit OBJECT_NAME(POSEIDON_LAZY(POSEIDON_REST, void OBJECT_FIELDS));

#endif // (0 OBJECT_FIELDS)

public:
	const char *get_table() const OVERRIDE;
	void generate_sql(::std::ostream &os_) const OVERRIDE;
	void fetch(const ::boost::shared_ptr<const ::Poseidon::Mysql::Connection> &conn_) OVERRIDE;
};

#ifdef MYSQL_OBJECT_EMIT_EXTERNAL_DEFINITIONS
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wshadow"

OBJECT_NAME::OBJECT_NAME()
	: ::Poseidon::Mysql::Object_base()

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

OBJECT_NAME::~OBJECT_NAME(){
	//
}

#undef FIELD_BOOLEAN
#undef FIELD_SIGNED
#undef FIELD_UNSIGNED
#undef FIELD_DOUBLE
#undef FIELD_STRING
#undef FIELD_DATETIME
#undef FIELD_UUID
#undef FIELD_BLOB

#define FIELD_BOOLEAN(id_)                + 1
#define FIELD_SIGNED(id_)                 + 1
#define FIELD_UNSIGNED(id_)               + 1
#define FIELD_DOUBLE(id_)                 + 1
#define FIELD_STRING(id_)                 + 1
#define FIELD_DATETIME(id_)               + 1
#define FIELD_UUID(id_)                   + 1
#define FIELD_BLOB(id_)                   + 1

#if (0 OBJECT_FIELDS)

#undef FIELD_BOOLEAN
#undef FIELD_SIGNED
#undef FIELD_UNSIGNED
#undef FIELD_DOUBLE
#undef FIELD_STRING
#undef FIELD_DATETIME
#undef FIELD_UUID
#undef FIELD_BLOB

#define FIELD_BOOLEAN(id_)                , bool param_##id_##_Zky_
#define FIELD_SIGNED(id_)                 , ::std::int64_t param_##id_##_Zky_
#define FIELD_UNSIGNED(id_)               , ::std::uint64_t param_##id_##_Zky_
#define FIELD_DOUBLE(id_)                 , double param_##id_##_Zky_
#define FIELD_STRING(id_)                 , ::std::string param_##id_##_Zky_
#define FIELD_DATETIME(id_)               , ::std::uint64_t param_##id_##_Zky_
#define FIELD_UUID(id_)                   , const ::Poseidon::Uuid & param_##id_##_Zky_
#define FIELD_BLOB(id_)                   , ::std::basic_string<unsigned char> param_##id_##_Zky_

OBJECT_NAME::OBJECT_NAME(POSEIDON_LAZY(POSEIDON_REST, void OBJECT_FIELDS))
	: ::Poseidon::Mysql::Object_base()

#undef FIELD_BOOLEAN
#undef FIELD_SIGNED
#undef FIELD_UNSIGNED
#undef FIELD_DOUBLE
#undef FIELD_STRING
#undef FIELD_DATETIME
#undef FIELD_UUID
#undef FIELD_BLOB

#define FIELD_BOOLEAN(id_)                , id_(this, param_##id_##_Zky_)
#define FIELD_SIGNED(id_)                 , id_(this, param_##id_##_Zky_)
#define FIELD_UNSIGNED(id_)               , id_(this, param_##id_##_Zky_)
#define FIELD_DOUBLE(id_)                 , id_(this, param_##id_##_Zky_)
#define FIELD_STRING(id_)                 , id_(this, STD_MOVE(param_##id_##_Zky_))
#define FIELD_DATETIME(id_)               , id_(this, param_##id_##_Zky_)
#define FIELD_UUID(id_)                   , id_(this, param_##id_##_Zky_)
#define FIELD_BLOB(id_)                   , id_(this, STD_MOVE(param_##id_##_Zky_))

	OBJECT_FIELDS
{ }

#endif // (0 OBJECT_FIELDS)

const char *OBJECT_NAME::get_table() const {
	return OBJECT_TABLE;
}
void OBJECT_NAME::generate_sql(::std::ostream &os_) const {
	POSEIDON_PROFILE_ME;

	const ::Poseidon::Recursive_mutex::Unique_lock lock_(m_mutex);

#undef FIELD_BOOLEAN
#undef FIELD_SIGNED
#undef FIELD_UNSIGNED
#undef FIELD_DOUBLE
#undef FIELD_STRING
#undef FIELD_DATETIME
#undef FIELD_UUID
#undef FIELD_BLOB

#define FIELD_BOOLEAN(id_)                os_ <<"`" POSEIDON_STRINGIFY(id_) "` = " <<id_.get() <<", ";
#define FIELD_SIGNED(id_)                 os_ <<"`" POSEIDON_STRINGIFY(id_) "` = " <<id_.get() <<", ";
#define FIELD_UNSIGNED(id_)               os_ <<"`" POSEIDON_STRINGIFY(id_) "` = " <<id_.get() <<", ";
#define FIELD_DOUBLE(id_)                 os_ <<"`" POSEIDON_STRINGIFY(id_) "` = " <<id_.get() <<", ";
#define FIELD_STRING(id_)                 os_ <<"`" POSEIDON_STRINGIFY(id_) "` = " << ::Poseidon::Mysql::String_escaper(id_.get()) <<", ";
#define FIELD_DATETIME(id_)               os_ <<"`" POSEIDON_STRINGIFY(id_) "` = " << ::Poseidon::Mysql::Date_time_formatter(id_.get()) <<", ";
#define FIELD_UUID(id_)                   os_ <<"`" POSEIDON_STRINGIFY(id_) "` = " << ::Poseidon::Mysql::Uuid_formatter(id_.get()) <<", ";
#define FIELD_BLOB(id_)                   os_ <<"`" POSEIDON_STRINGIFY(id_) "` = " << ::Poseidon::Mysql::String_escaper(id_.get()) <<", ";

	OBJECT_FIELDS
}
void OBJECT_NAME::fetch(const ::boost::shared_ptr<const ::Poseidon::Mysql::Connection> &conn_){
	POSEIDON_PROFILE_ME;

	const ::Poseidon::Recursive_mutex::Unique_lock lock_(m_mutex);

#undef FIELD_BOOLEAN
#undef FIELD_SIGNED
#undef FIELD_UNSIGNED
#undef FIELD_DOUBLE
#undef FIELD_STRING
#undef FIELD_DATETIME
#undef FIELD_UUID
#undef FIELD_BLOB

#define FIELD_BOOLEAN(id_)                id_.set(conn_->get_boolean  ( POSEIDON_STRINGIFY(id_) ), false);
#define FIELD_SIGNED(id_)                 id_.set(conn_->get_signed   ( POSEIDON_STRINGIFY(id_) ), false);
#define FIELD_UNSIGNED(id_)               id_.set(conn_->get_unsigned ( POSEIDON_STRINGIFY(id_) ), false);
#define FIELD_DOUBLE(id_)                 id_.set(conn_->get_double   ( POSEIDON_STRINGIFY(id_) ), false);
#define FIELD_STRING(id_)                 id_.set(conn_->get_string   ( POSEIDON_STRINGIFY(id_) ), false);
#define FIELD_DATETIME(id_)               id_.set(conn_->get_datetime ( POSEIDON_STRINGIFY(id_) ), false);
#define FIELD_UUID(id_)                   id_.set(conn_->get_uuid     ( POSEIDON_STRINGIFY(id_) ), false);
#define FIELD_BLOB(id_)                   id_.set(conn_->get_blob     ( POSEIDON_STRINGIFY(id_) ), false);

	OBJECT_FIELDS
}

#pragma GCC diagnostic pop
#endif // MYSQL_OBJECT_EMIT_EXTERNAL_DEFINITIONS

#undef OBJECT_NAME
#undef OBJECT_FIELDS
