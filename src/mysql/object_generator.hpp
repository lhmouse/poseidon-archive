// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2017, LH_Mouse. All wrongs reserved.

#ifndef OBJECT_NAME
#  error OBJECT_NAME is undefined.
#endif

#ifndef OBJECT_FIELDS
#  error OBJECT_FIELDS is undefined.
#endif

#ifndef POSEIDON_MYSQL_OBJECT_BASE_HPP_
#  error Please #include <poseidon/mysql/object_base.hpp> first.
#endif

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wshadow"

class OBJECT_NAME : public ::Poseidon::MySql::ObjectBase {
public:
	static ::boost::shared_ptr< ::Poseidon::MySql::ObjectBase> create(){
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

#define FIELD_BOOLEAN(id_)                ::Poseidon::MySql::ObjectBase::Field<bool> id_;
#define FIELD_SIGNED(id_)                 ::Poseidon::MySql::ObjectBase::Field< ::boost::int64_t> id_;
#define FIELD_UNSIGNED(id_)               ::Poseidon::MySql::ObjectBase::Field< ::boost::uint64_t> id_;
#define FIELD_DOUBLE(id_)                 ::Poseidon::MySql::ObjectBase::Field<double> id_;
#define FIELD_STRING(id_)                 ::Poseidon::MySql::ObjectBase::Field< ::std::string> id_;
#define FIELD_DATETIME(id_)               ::Poseidon::MySql::ObjectBase::Field< ::boost::uint64_t> id_;
#define FIELD_UUID(id_)                   ::Poseidon::MySql::ObjectBase::Field< ::Poseidon::Uuid> id_;
#define FIELD_BLOB(id_)                   ::Poseidon::MySql::ObjectBase::Field< ::std::basic_string<unsigned char> > id_;

	OBJECT_FIELDS

public:
	OBJECT_NAME()
		: ::Poseidon::MySql::ObjectBase()

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

	explicit OBJECT_NAME(STRIP_FIRST(void OBJECT_FIELDS))
		: ::Poseidon::MySql::ObjectBase()

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
	~OBJECT_NAME() OVERRIDE;

public:
	const char *get_table() const OVERRIDE {
		return TOKEN_TO_STR(OBJECT_NAME);
	}

	void generate_sql(::std::ostream &os_) const OVERRIDE {
		static CONSTEXPR const char delims_[2][4] = { "", ", " };
		bool flag_ = false;

#undef FIELD_BOOLEAN
#undef FIELD_SIGNED
#undef FIELD_UNSIGNED
#undef FIELD_DOUBLE
#undef FIELD_STRING
#undef FIELD_DATETIME
#undef FIELD_UUID
#undef FIELD_BLOB

#define FIELD_BOOLEAN(id_)                os_ <<delims_[flag_++] <<"`" TOKEN_TO_STR(id_) "` = " <<id_;
#define FIELD_SIGNED(id_)                 os_ <<delims_[flag_++] <<"`" TOKEN_TO_STR(id_) "` = " <<id_;
#define FIELD_UNSIGNED(id_)               os_ <<delims_[flag_++] <<"`" TOKEN_TO_STR(id_) "` = " <<id_;
#define FIELD_DOUBLE(id_)                 os_ <<delims_[flag_++] <<"`" TOKEN_TO_STR(id_) "` = " <<id_;
#define FIELD_STRING(id_)                 os_ <<delims_[flag_++] <<"`" TOKEN_TO_STR(id_) "` = " << ::Poseidon::MySql::StringEscaper(id_);
#define FIELD_DATETIME(id_)               os_ <<delims_[flag_++] <<"`" TOKEN_TO_STR(id_) "` = " << ::Poseidon::MySql::DateTimeFormatter(id_);
#define FIELD_UUID(id_)                   os_ <<delims_[flag_++] <<"`" TOKEN_TO_STR(id_) "` = " << ::Poseidon::MySql::UuidFormatter(id_);
#define FIELD_BLOB(id_)                   os_ <<delims_[flag_++] <<"`" TOKEN_TO_STR(id_) "` = " << ::Poseidon::MySql::StringEscaper(id_);

		OBJECT_FIELDS
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

#define FIELD_BOOLEAN(id_)                id_.set(conn_->get_signed   ( TOKEN_TO_STR(id_) ), false);
#define FIELD_SIGNED(id_)                 id_.set(conn_->get_signed   ( TOKEN_TO_STR(id_) ), false);
#define FIELD_UNSIGNED(id_)               id_.set(conn_->get_unsigned ( TOKEN_TO_STR(id_) ), false);
#define FIELD_DOUBLE(id_)                 id_.set(conn_->get_double   ( TOKEN_TO_STR(id_) ), false);
#define FIELD_STRING(id_)                 id_.set(conn_->get_string   ( TOKEN_TO_STR(id_) ), false);
#define FIELD_DATETIME(id_)               id_.set(conn_->get_datetime ( TOKEN_TO_STR(id_) ), false);
#define FIELD_UUID(id_)                   id_.set(conn_->get_uuid     ( TOKEN_TO_STR(id_) ), false);
#define FIELD_BLOB(id_)                   id_.set(conn_->get_blob     ( TOKEN_TO_STR(id_) ), false);

		OBJECT_FIELDS
	}
};

#ifdef OBJECT_DEFINE_RTTI
OBJECT_NAME::~OBJECT_NAME(){ }
#endif

#pragma GCC diagnostic pop

#undef OBJECT_NAME
#undef OBJECT_FIELDS
