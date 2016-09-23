// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2016, LH_Mouse. All wrongs reserved.

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

private:

#undef FIELD_BOOLEAN
#undef FIELD_TINYINT
#undef FIELD_TINYINT_UNSIGNED
#undef FIELD_SMALLINT
#undef FIELD_SMALLINT_UNSIGNED
#undef FIELD_INTEGER
#undef FIELD_INTEGER_UNSIGNED
#undef FIELD_BIGINT
#undef FIELD_BIGINT_UNSIGNED
#undef FIELD_DOUBLE
#undef FIELD_STRING
#undef FIELD_DATETIME
#undef FIELD_UUID

#define FIELD_BOOLEAN(name_)                bool name_;
#define FIELD_TINYINT(name_)                ::boost::int8_t name_;
#define FIELD_TINYINT_UNSIGNED(name_)       ::boost::uint8_t name_;
#define FIELD_SMALLINT(name_)               ::boost::int16_t name_;
#define FIELD_SMALLINT_UNSIGNED(name_)      ::boost::uint16_t name_;
#define FIELD_INTEGER(name_)                ::boost::int32_t name_;
#define FIELD_INTEGER_UNSIGNED(name_)       ::boost::uint32_t name_;
#define FIELD_BIGINT(name_)                 ::boost::int64_t name_;
#define FIELD_BIGINT_UNSIGNED(name_)        ::boost::uint64_t name_;
#define FIELD_DOUBLE(name_)                 double name_;
#define FIELD_STRING(name_)                 ::std::string name_;
#define FIELD_DATETIME(name_)               ::boost::uint64_t name_;
#define FIELD_UUID(name_)                   ::Poseidon::Uuid name_;

	MYSQL_OBJECT_FIELDS

public:
	MYSQL_OBJECT_NAME()
		: ::Poseidon::MySql::ObjectBase()

#undef FIELD_BOOLEAN
#undef FIELD_TINYINT
#undef FIELD_TINYINT_UNSIGNED
#undef FIELD_SMALLINT
#undef FIELD_SMALLINT_UNSIGNED
#undef FIELD_INTEGER
#undef FIELD_INTEGER_UNSIGNED
#undef FIELD_BIGINT
#undef FIELD_BIGINT_UNSIGNED
#undef FIELD_DOUBLE
#undef FIELD_STRING
#undef FIELD_DATETIME
#undef FIELD_UUID

#define FIELD_BOOLEAN(name_)                , name_()
#define FIELD_TINYINT(name_)                , name_()
#define FIELD_TINYINT_UNSIGNED(name_)       , name_()
#define FIELD_SMALLINT(name_)               , name_()
#define FIELD_SMALLINT_UNSIGNED(name_)      , name_()
#define FIELD_INTEGER(name_)                , name_()
#define FIELD_INTEGER_UNSIGNED(name_)       , name_()
#define FIELD_BIGINT(name_)                 , name_()
#define FIELD_BIGINT_UNSIGNED(name_)        , name_()
#define FIELD_DOUBLE(name_)                 , name_()
#define FIELD_STRING(name_)                 , name_()
#define FIELD_DATETIME(name_)               , name_()
#define FIELD_UUID(name_)                   , name_()

		MYSQL_OBJECT_FIELDS
	{
	}

#undef FIELD_BOOLEAN
#undef FIELD_TINYINT
#undef FIELD_TINYINT_UNSIGNED
#undef FIELD_SMALLINT
#undef FIELD_SMALLINT_UNSIGNED
#undef FIELD_INTEGER
#undef FIELD_INTEGER_UNSIGNED
#undef FIELD_BIGINT
#undef FIELD_BIGINT_UNSIGNED
#undef FIELD_DOUBLE
#undef FIELD_STRING
#undef FIELD_DATETIME
#undef FIELD_UUID

#define FIELD_BOOLEAN(name_)                , bool name_ ## X_
#define FIELD_TINYINT(name_)                , ::boost::int8_t name_ ## X_
#define FIELD_TINYINT_UNSIGNED(name_)       , ::boost::uint8_t name_ ## X_
#define FIELD_SMALLINT(name_)               , ::boost::int16_t name_ ## X_
#define FIELD_SMALLINT_UNSIGNED(name_)      , ::boost::uint16_t name_ ## X_
#define FIELD_INTEGER(name_)                , ::boost::int32_t name_ ## X_
#define FIELD_INTEGER_UNSIGNED(name_)       , ::boost::uint32_t name_ ## X_
#define FIELD_BIGINT(name_)                 , ::boost::int64_t name_ ## X_
#define FIELD_BIGINT_UNSIGNED(name_)        , ::boost::uint64_t name_ ## X_
#define FIELD_DOUBLE(name_)                 , double name_ ## X_
#define FIELD_STRING(name_)                 , ::std::string name_ ## X_
#define FIELD_DATETIME(name_)               , ::boost::uint64_t name_ ## X_
#define FIELD_UUID(name_)                   , const ::Poseidon::Uuid & name_ ## X_

	explicit MYSQL_OBJECT_NAME(STRIP_FIRST(void MYSQL_OBJECT_FIELDS))
		: ::Poseidon::MySql::ObjectBase()

#undef FIELD_BOOLEAN
#undef FIELD_TINYINT
#undef FIELD_TINYINT_UNSIGNED
#undef FIELD_SMALLINT
#undef FIELD_SMALLINT_UNSIGNED
#undef FIELD_INTEGER
#undef FIELD_INTEGER_UNSIGNED
#undef FIELD_BIGINT
#undef FIELD_BIGINT_UNSIGNED
#undef FIELD_DOUBLE
#undef FIELD_STRING
#undef FIELD_DATETIME
#undef FIELD_UUID

#define FIELD_BOOLEAN(name_)                , name_(name_ ## X_)
#define FIELD_TINYINT(name_)                , name_(name_ ## X_)
#define FIELD_TINYINT_UNSIGNED(name_)       , name_(name_ ## X_)
#define FIELD_SMALLINT(name_)               , name_(name_ ## X_)
#define FIELD_SMALLINT_UNSIGNED(name_)      , name_(name_ ## X_)
#define FIELD_INTEGER(name_)                , name_(name_ ## X_)
#define FIELD_INTEGER_UNSIGNED(name_)       , name_(name_ ## X_)
#define FIELD_BIGINT(name_)                 , name_(name_ ## X_)
#define FIELD_BIGINT_UNSIGNED(name_)        , name_(name_ ## X_)
#define FIELD_DOUBLE(name_)                 , name_(name_ ## X_)
#define FIELD_STRING(name_)                 , name_(STD_MOVE(name_ ## X_))
#define FIELD_DATETIME(name_)               , name_(name_ ## X_)
#define FIELD_UUID(name_)                   , name_(name_ ## X_)

		MYSQL_OBJECT_FIELDS
	{
		::Poseidon::atomic_fence(::Poseidon::ATOMIC_RELEASE);
	}

public:

#undef FIELD_BOOLEAN
#undef FIELD_TINYINT
#undef FIELD_TINYINT_UNSIGNED
#undef FIELD_SMALLINT
#undef FIELD_SMALLINT_UNSIGNED
#undef FIELD_INTEGER
#undef FIELD_INTEGER_UNSIGNED
#undef FIELD_BIGINT
#undef FIELD_BIGINT_UNSIGNED
#undef FIELD_DOUBLE
#undef FIELD_STRING
#undef FIELD_DATETIME
#undef FIELD_UUID

#define FIELD_BOOLEAN(name_)                const bool & unlocked_get_ ## name_() const {	\
                                            	return name_;	\
                                            }	\
                                            bool get_ ## name_() const {	\
                                            	return ::Poseidon::atomic_load(name_, ::Poseidon::ATOMIC_CONSUME);	\
                                            }	\
                                            void set_ ## name_(bool val_, bool invalidates_ = true){	\
                                            	::Poseidon::atomic_store(name_, val_, ::Poseidon::ATOMIC_RELEASE);	\
                                            	if(invalidates_){ invalidate(); }	\
                                            }
#define FIELD_TINYINT(name_)                const ::boost::int8_t & unlocked_get_ ## name_() const {	\
                                            	return name_;	\
                                            }	\
                                            ::boost::int8_t get_ ## name_() const {	\
                                            	return ::Poseidon::atomic_load(name_, ::Poseidon::ATOMIC_CONSUME);	\
                                            }	\
                                            void set_ ## name_(::boost::int8_t val_, bool invalidates_ = true){	\
                                            	::Poseidon::atomic_store(name_, val_, ::Poseidon::ATOMIC_RELEASE);	\
                                            	if(invalidates_){ invalidate(); }	\
                                            }
#define FIELD_TINYINT_UNSIGNED(name_)       const ::boost::uint8_t & unlocked_get_ ## name_() const {	\
                                            	return name_;	\
                                            }	\
                                            ::boost::uint8_t get_ ## name_() const {	\
                                            	return ::Poseidon::atomic_load(name_, ::Poseidon::ATOMIC_CONSUME);	\
                                            }	\
                                            void set_ ## name_(::boost::uint8_t val_, bool invalidates_ = true){	\
                                            	::Poseidon::atomic_store(name_, val_, ::Poseidon::ATOMIC_RELEASE);	\
                                            	if(invalidates_){ invalidate(); }	\
                                            }
#define FIELD_SMALLINT(name_)               const ::boost::int16_t & unlocked_get_ ## name_() const {	\
                                            	return name_;	\
                                            }	\
                                            ::boost::int16_t get_ ## name_() const {	\
                                            	return ::Poseidon::atomic_load(name_, ::Poseidon::ATOMIC_CONSUME);	\
                                            }	\
                                            void set_ ## name_(::boost::int16_t val_, bool invalidates_ = true){	\
                                            	::Poseidon::atomic_store(name_, val_, ::Poseidon::ATOMIC_RELEASE);	\
                                            	if(invalidates_){ invalidate(); }	\
                                            }
#define FIELD_SMALLINT_UNSIGNED(name_)      const ::boost::uint16_t & unlocked_get_ ## name_() const {	\
                                            	return name_;	\
                                            }	\
                                            ::boost::uint16_t get_ ## name_() const {	\
                                            	return ::Poseidon::atomic_load(name_, ::Poseidon::ATOMIC_CONSUME);	\
                                            }	\
                                            void set_ ## name_(::boost::uint16_t val_, bool invalidates_ = true){	\
                                            	::Poseidon::atomic_store(name_, val_, ::Poseidon::ATOMIC_RELEASE);	\
                                            	if(invalidates_){ invalidate(); }	\
                                            }
#define FIELD_INTEGER(name_)                const ::boost::int32_t & unlocked_get_ ## name_() const {	\
                                            	return name_;	\
                                            }	\
                                            ::boost::int32_t get_ ## name_() const {	\
                                            	return ::Poseidon::atomic_load(name_, ::Poseidon::ATOMIC_CONSUME);	\
                                            }	\
                                            void set_ ## name_(::boost::int32_t val_, bool invalidates_ = true){	\
                                            	::Poseidon::atomic_store(name_, val_, ::Poseidon::ATOMIC_RELEASE);	\
                                            	if(invalidates_){ invalidate(); }	\
                                            }
#define FIELD_INTEGER_UNSIGNED(name_)       const ::boost::uint32_t & unlocked_get_ ## name_() const {	\
                                            	return name_;	\
                                            }	\
                                            ::boost::uint32_t get_ ## name_() const {	\
                                            	return ::Poseidon::atomic_load(name_, ::Poseidon::ATOMIC_CONSUME);	\
                                            }	\
                                            void set_ ## name_(::boost::uint32_t val_, bool invalidates_ = true){	\
                                            	::Poseidon::atomic_store(name_, val_, ::Poseidon::ATOMIC_RELEASE);	\
                                            	if(invalidates_){ invalidate(); }	\
                                            }
#define FIELD_BIGINT(name_)                 const ::boost::int64_t & unlocked_get_ ## name_() const {	\
                                            	return name_;	\
                                            }	\
                                            ::boost::int64_t get_ ## name_() const {	\
                                            	return ::Poseidon::atomic_load(name_, ::Poseidon::ATOMIC_CONSUME);	\
                                            }	\
                                            void set_ ## name_(::boost::int64_t val_, bool invalidates_ = true){	\
                                            	::Poseidon::atomic_store(name_, val_, ::Poseidon::ATOMIC_RELEASE);	\
                                            	if(invalidates_){ invalidate(); }	\
                                            }
#define FIELD_BIGINT_UNSIGNED(name_)        const ::boost::uint64_t & unlocked_get_ ## name_() const {	\
                                            	return name_;	\
                                            }	\
                                            ::boost::uint64_t get_ ## name_() const {	\
                                            	return ::Poseidon::atomic_load(name_, ::Poseidon::ATOMIC_CONSUME);	\
                                            }	\
                                            void set_ ## name_(::boost::uint64_t val_, bool invalidates_ = true){	\
                                            	::Poseidon::atomic_store(name_, val_, ::Poseidon::ATOMIC_RELEASE);	\
                                            	if(invalidates_){ invalidate(); }	\
                                            }
#define FIELD_DOUBLE(name_)                 const double & unlocked_get_ ## name_() const {	\
                                            	return name_;	\
                                            }	\
                                            double get_ ## name_() const {	\
                                            	const ::Poseidon::Mutex::UniqueLock lock_(m_mutex);	\
                                            	return name_;	\
                                            }	\
                                            void set_ ## name_(double val_, bool invalidates_ = true){	\
                                            	{ const ::Poseidon::Mutex::UniqueLock lock_(m_mutex);	\
                                            	  name_ = val_; }	\
                                            	if(invalidates_){ invalidate(); }	\
                                            }
#define FIELD_STRING(name_)                 const ::std::string & unlocked_get_ ## name_() const {	\
                                            	return name_;	\
                                            }	\
                                            ::std::string get_ ## name_() const {	\
                                            	const ::Poseidon::Mutex::UniqueLock lock_(m_mutex);	\
                                            	return name_;	\
                                            }	\
                                            void set_ ## name_(::std::string val_, bool invalidates_ = true){	\
                                            	{ const ::Poseidon::Mutex::UniqueLock lock_(m_mutex);	\
                                            	  name_.swap(val_); }	\
                                            	if(invalidates_){ invalidate(); }	\
                                            }
#define FIELD_DATETIME(name_)               const ::boost::uint64_t & unlocked_get_ ## name_() const {	\
                                            	return name_;	\
                                            }	\
                                            ::boost::uint64_t get_ ## name_() const {	\
                                            	return ::Poseidon::atomic_load(name_, ::Poseidon::ATOMIC_CONSUME);	\
                                            }	\
                                            void set_ ## name_(::boost::uint64_t val_, bool invalidates_ = true){	\
                                            	::Poseidon::atomic_store(name_, val_, ::Poseidon::ATOMIC_RELEASE);	\
                                            	if(invalidates_){ invalidate(); }	\
                                            }
#define FIELD_UUID(name_)                   const ::Poseidon::Uuid & unlocked_get_ ## name_() const {	\
                                            	return name_;	\
                                            }	\
                                            Poseidon::Uuid get_ ## name_() const {	\
                                            	const ::Poseidon::Mutex::UniqueLock lock_(m_mutex);	\
                                            	return name_;	\
                                            }	\
                                            void set_ ## name_(const ::Poseidon::Uuid &val_, bool invalidates_ = true){	\
                                            	{ const ::Poseidon::Mutex::UniqueLock lock_(m_mutex);	\
                                            	  name_ = val_; }	\
                                            	if(invalidates_){ invalidate(); }	\
                                            }

	MYSQL_OBJECT_FIELDS

	const char *get_table_name() const OVERRIDE {
		return TOKEN_TO_STR(MYSQL_OBJECT_NAME);
	}

	void generate_sql(std::ostream &os_) const OVERRIDE {

#undef FIELD_BOOLEAN
#undef FIELD_TINYINT
#undef FIELD_TINYINT_UNSIGNED
#undef FIELD_SMALLINT
#undef FIELD_SMALLINT_UNSIGNED
#undef FIELD_INTEGER
#undef FIELD_INTEGER_UNSIGNED
#undef FIELD_BIGINT
#undef FIELD_BIGINT_UNSIGNED
#undef FIELD_DOUBLE
#undef FIELD_STRING
#undef FIELD_DATETIME
#undef FIELD_UUID

#define FIELD_BOOLEAN(name_)                (void)(os_ <<", "),	\
                                            (void)(os_ <<"`" TOKEN_TO_STR(name_) "` = " <<static_cast<long>(get_ ## name_())),
#define FIELD_TINYINT(name_)                (void)(os_ <<", "),	\
                                            (void)(os_ <<"`" TOKEN_TO_STR(name_) "` = " <<static_cast<long>(get_ ## name_())),
#define FIELD_TINYINT_UNSIGNED(name_)       (void)(os_ <<", "),	\
                                            (void)(os_ <<"`" TOKEN_TO_STR(name_) "` = " <<static_cast<unsigned long>(get_ ## name_())),
#define FIELD_SMALLINT(name_)               (void)(os_ <<", "),	\
                                            (void)(os_ <<"`" TOKEN_TO_STR(name_) "` = " <<static_cast<long>(get_ ## name_())),
#define FIELD_SMALLINT_UNSIGNED(name_)      (void)(os_ <<", "),	\
                                            (void)(os_ <<"`" TOKEN_TO_STR(name_) "` = " <<static_cast<unsigned long>(get_ ## name_())),
#define FIELD_INTEGER(name_)                (void)(os_ <<", "),	\
                                            (void)(os_ <<"`" TOKEN_TO_STR(name_) "` = " <<static_cast<long>(get_ ## name_())),
#define FIELD_INTEGER_UNSIGNED(name_)       (void)(os_ <<", "),	\
                                            (void)(os_ <<"`" TOKEN_TO_STR(name_) "` = " <<static_cast<unsigned long>(get_ ## name_())),
#define FIELD_BIGINT(name_)                 (void)(os_ <<", "),	\
                                            (void)(os_ <<"`" TOKEN_TO_STR(name_) "` = " <<static_cast< ::boost::int64_t>(get_ ## name_())),
#define FIELD_BIGINT_UNSIGNED(name_)        (void)(os_ <<", "),	\
                                            (void)(os_ <<"`" TOKEN_TO_STR(name_) "` = " <<static_cast< ::boost::uint64_t>(get_ ## name_())),
#define FIELD_DOUBLE(name_)                 (void)(os_ <<", "),	\
                                            (void)(os_ <<"`" TOKEN_TO_STR(name_) "` = " <<static_cast<double>(get_ ## name_())),
#define FIELD_STRING(name_)                 (void)(os_ <<", "),	\
                                            (void)(os_ <<"`" TOKEN_TO_STR(name_) "` = " << ::Poseidon::MySql::StringEscaper(get_ ## name_())),
#define FIELD_DATETIME(name_)               (void)(os_ <<", "),	\
                                            (void)(os_ <<"`" TOKEN_TO_STR(name_) "` = " << ::Poseidon::MySql::DateTimeFormatter(get_ ## name_())),
#define FIELD_UUID(name_)                   (void)(os_ <<", "),	\
                                            (void)(os_ <<"`" TOKEN_TO_STR(name_) "` = " << ::Poseidon::MySql::UuidFormatter(get_ ## name_())),

		STRIP_FIRST(MYSQL_OBJECT_FIELDS) (void)0;
	}
	void fetch(const boost::shared_ptr<const ::Poseidon::MySql::Connection> &conn_) OVERRIDE {

#undef FIELD_BOOLEAN
#undef FIELD_TINYINT
#undef FIELD_TINYINT_UNSIGNED
#undef FIELD_SMALLINT
#undef FIELD_SMALLINT_UNSIGNED
#undef FIELD_INTEGER
#undef FIELD_INTEGER_UNSIGNED
#undef FIELD_BIGINT
#undef FIELD_BIGINT_UNSIGNED
#undef FIELD_DOUBLE
#undef FIELD_STRING
#undef FIELD_DATETIME
#undef FIELD_UUID

#define FIELD_BOOLEAN(name_)                set_ ## name_(conn_->get_signed  ( TOKEN_TO_STR(name_) ), false);
#define FIELD_TINYINT(name_)                set_ ## name_(conn_->get_signed  ( TOKEN_TO_STR(name_) ), false);
#define FIELD_TINYINT_UNSIGNED(name_)       set_ ## name_(conn_->get_unsigned( TOKEN_TO_STR(name_) ), false);
#define FIELD_SMALLINT(name_)               set_ ## name_(conn_->get_signed  ( TOKEN_TO_STR(name_) ), false);
#define FIELD_SMALLINT_UNSIGNED(name_)      set_ ## name_(conn_->get_unsigned( TOKEN_TO_STR(name_) ), false);
#define FIELD_INTEGER(name_)                set_ ## name_(conn_->get_signed  ( TOKEN_TO_STR(name_) ), false);
#define FIELD_INTEGER_UNSIGNED(name_)       set_ ## name_(conn_->get_unsigned( TOKEN_TO_STR(name_) ), false);
#define FIELD_BIGINT(name_)                 set_ ## name_(conn_->get_signed  ( TOKEN_TO_STR(name_) ), false);
#define FIELD_BIGINT_UNSIGNED(name_)        set_ ## name_(conn_->get_unsigned( TOKEN_TO_STR(name_) ), false);
#define FIELD_DOUBLE(name_)                 set_ ## name_(conn_->get_double  ( TOKEN_TO_STR(name_) ), false);
#define FIELD_STRING(name_)                 set_ ## name_(conn_->get_string  ( TOKEN_TO_STR(name_) ), false);
#define FIELD_DATETIME(name_)               set_ ## name_(conn_->get_datetime( TOKEN_TO_STR(name_) ), false);
#define FIELD_UUID(name_)                   set_ ## name_(conn_->get_uuid    ( TOKEN_TO_STR(name_) ), false);

		MYSQL_OBJECT_FIELDS
	}
};

#pragma GCC pop_options

#undef MYSQL_OBJECT_NAME
#undef MYSQL_OBJECT_FIELDS
