// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2017, LH_Mouse. All wrongs reserved.

#ifndef MYSQL_OBJECT_NAME
#   error MYSQL_OBJECT_NAME is undefined.
#endif

#ifndef MYSQL_OBJECT_FIELDS
#   error MYSQL_OBJECT_FIELDS is undefined.
#endif

#ifndef POSEIDON_MYSQL_OBJECT_BASE_HPP_
#   error Please #include <poseidon/mysql/object_base.hpp> first.
#endif

#pragma GCC push_options
#pragma GCC diagnostic ignored "-Wshadow"

class MYSQL_OBJECT_NAME : public ::Poseidon::MySql::ObjectBase {
public:
	static ::boost::shared_ptr< ::Poseidon::MySql::ObjectBase> create(){
		return ::boost::make_shared<MYSQL_OBJECT_NAME>();
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

#define FIELD_BOOLEAN(name_)                ::Poseidon::MySql::ObjectBase::Field<bool> name_;
#define FIELD_SIGNED(name_)                 ::Poseidon::MySql::ObjectBase::Field< ::boost::int64_t> name_;
#define FIELD_UNSIGNED(name_)               ::Poseidon::MySql::ObjectBase::Field< ::boost::uint64_t> name_;
#define FIELD_DOUBLE(name_)                 ::Poseidon::MySql::ObjectBase::Field<double> name_;
#define FIELD_STRING(name_)                 ::Poseidon::MySql::ObjectBase::Field< ::std::string> name_;
#define FIELD_DATETIME(name_)               ::Poseidon::MySql::ObjectBase::Field< ::boost::uint64_t> name_;
#define FIELD_UUID(name_)                   ::Poseidon::MySql::ObjectBase::Field< ::Poseidon::Uuid> name_;
#define FIELD_BLOB(name_)                   ::Poseidon::MySql::ObjectBase::Field< ::std::string> name_;

	MYSQL_OBJECT_FIELDS

public:
	MYSQL_OBJECT_NAME()
		: ::Poseidon::MySql::ObjectBase()

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

		MYSQL_OBJECT_FIELDS
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

	explicit MYSQL_OBJECT_NAME(STRIP_FIRST(void MYSQL_OBJECT_FIELDS))
		: ::Poseidon::MySql::ObjectBase()

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

		MYSQL_OBJECT_FIELDS
	{
		::Poseidon::atomic_fence(::Poseidon::ATOMIC_RELEASE);
	}

public:
	const char *get_table() const OVERRIDE {
		return TOKEN_TO_STR(MYSQL_OBJECT_NAME);
	}

	void generate_sql(::std::ostream &os_) const OVERRIDE {

#undef FIELD_BOOLEAN
#undef FIELD_SIGNED
#undef FIELD_UNSIGNED
#undef FIELD_DOUBLE
#undef FIELD_STRING
#undef FIELD_DATETIME
#undef FIELD_UUID
#undef FIELD_BLOB

#define FIELD_BOOLEAN(name_)                (void)(os_ <<", "),	\
                                            (void)(os_ <<"`" TOKEN_TO_STR(name_) "` = " <<name_),
#define FIELD_SIGNED(name_)                 (void)(os_ <<", "),	\
                                            (void)(os_ <<"`" TOKEN_TO_STR(name_) "` = " <<name_),
#define FIELD_UNSIGNED(name_)               (void)(os_ <<", "),	\
                                            (void)(os_ <<"`" TOKEN_TO_STR(name_) "` = " <<name_),
#define FIELD_DOUBLE(name_)                 (void)(os_ <<", "),	\
                                            (void)(os_ <<"`" TOKEN_TO_STR(name_) "` = " <<name_),
#define FIELD_STRING(name_)                 (void)(os_ <<", "),	\
                                            (void)(os_ <<"`" TOKEN_TO_STR(name_) "` = " << ::Poseidon::MySql::StringEscaper(name_)),
#define FIELD_DATETIME(name_)               (void)(os_ <<", "),	\
                                            (void)(os_ <<"`" TOKEN_TO_STR(name_) "` = " << ::Poseidon::MySql::DateTimeFormatter(name_)),
#define FIELD_UUID(name_)                   (void)(os_ <<", "),	\
                                            (void)(os_ <<"`" TOKEN_TO_STR(name_) "` = " << ::Poseidon::MySql::UuidFormatter(name_)),
#define FIELD_BLOB(name_)                   (void)(os_ <<", "),	\
                                            (void)(os_ <<"`" TOKEN_TO_STR(name_) "` = " << ::Poseidon::MySql::StringEscaper(name_)),

		STRIP_FIRST(MYSQL_OBJECT_FIELDS) (void)0;
	}
	void fetch(const ::boost::shared_ptr<const ::Poseidon::MySql::Connection> &conn_) OVERRIDE {

#undef FIELD_BOOLEAN
#undef FIELD_SIGNED
#undef FIELD_UNSIGNED
#undef FIELD_DOUBLE
#undef FIELD_STRING
#undef FIELD_DATETIME
#undef FIELD_UUID
#undef FIELD_BLOB

#define FIELD_BOOLEAN(name_)                name_.set(conn_->get_signed   ( TOKEN_TO_STR(name_) ), false);
#define FIELD_SIGNED(name_)                 name_.set(conn_->get_signed   ( TOKEN_TO_STR(name_) ), false);
#define FIELD_UNSIGNED(name_)               name_.set(conn_->get_unsigned ( TOKEN_TO_STR(name_) ), false);
#define FIELD_DOUBLE(name_)                 name_.set(conn_->get_double   ( TOKEN_TO_STR(name_) ), false);
#define FIELD_STRING(name_)                 name_.set(conn_->get_string   ( TOKEN_TO_STR(name_) ), false);
#define FIELD_DATETIME(name_)               name_.set(conn_->get_datetime ( TOKEN_TO_STR(name_) ), false);
#define FIELD_UUID(name_)                   name_.set(conn_->get_uuid     ( TOKEN_TO_STR(name_) ), false);
#define FIELD_BLOB(name_)                   name_.set(conn_->get_string   ( TOKEN_TO_STR(name_) ), false);

		MYSQL_OBJECT_FIELDS
	}
};

#pragma GCC pop_options

#undef MYSQL_OBJECT_NAME
#undef MYSQL_OBJECT_FIELDS
