// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

#ifndef MYSQL_OBJECT_NAME
#	error MYSQL_OBJECT_NAME is undefined.
#endif

#ifndef MYSQL_OBJECT_FIELDS
#	error MYSQL_OBJECT_FIELDS is undefined.
#endif

#ifndef POSEIDON_MYSQL_OBJECT_BASE_HPP_
#	error Please #include <poseidon/mysql/object_base.hpp> first.
#endif

class MYSQL_OBJECT_NAME : public ::Poseidon::MySql::ObjectBase {
public:
	static ::boost::shared_ptr< ::Poseidon::MySql::ObjectBase> create(){
		return ::boost::make_shared<MYSQL_OBJECT_NAME>();
	}

	static void batchLoad(std::string query, ::Poseidon::MySql::BatchAsyncLoadCallback callback,
		::Poseidon::MySql::ExceptionCallback except = ::Poseidon::MySql::ExceptionCallback())
	{
		::Poseidon::MySql::ObjectBase::batchLoad(&create,
			TOKEN_TO_STR(MYSQL_OBJECT_NAME), STD_MOVE(query),
			STD_MOVE(callback), STD_MOVE(except));
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
#undef FIELD_STRING
#undef FIELD_DATETIME

#define FIELD_BOOLEAN(name_)				volatile bool name_;
#define FIELD_TINYINT(name_)				volatile ::boost::int8_t name_;
#define FIELD_TINYINT_UNSIGNED(name_)		volatile ::boost::uint8_t name_;
#define FIELD_SMALLINT(name_)				volatile ::boost::int16_t name_;
#define FIELD_SMALLINT_UNSIGNED(name_)		volatile ::boost::uint16_t name_;
#define FIELD_INTEGER(name_)				volatile ::boost::int32_t name_;
#define FIELD_INTEGER_UNSIGNED(name_)		volatile ::boost::uint32_t name_;
#define FIELD_BIGINT(name_)					volatile ::boost::int64_t name_;
#define FIELD_BIGINT_UNSIGNED(name_)		volatile ::boost::uint64_t name_;
#define FIELD_STRING(name_)					::std::string name_;
#define FIELD_DATETIME(name_)				volatile ::boost::uint64_t name_;

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
#undef FIELD_STRING
#undef FIELD_DATETIME

#define FIELD_BOOLEAN(name_)				, name_()
#define FIELD_TINYINT(name_)				, name_()
#define FIELD_TINYINT_UNSIGNED(name_)		, name_()
#define FIELD_SMALLINT(name_)				, name_()
#define FIELD_SMALLINT_UNSIGNED(name_)		, name_()
#define FIELD_INTEGER(name_)				, name_()
#define FIELD_INTEGER_UNSIGNED(name_)		, name_()
#define FIELD_BIGINT(name_)					, name_()
#define FIELD_BIGINT_UNSIGNED(name_)		, name_()
#define FIELD_STRING(name_)					, name_()
#define FIELD_DATETIME(name_)				, name_()

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
#undef FIELD_STRING
#undef FIELD_DATETIME

#define FIELD_BOOLEAN(name_)				, bool name_ ## X_
#define FIELD_TINYINT(name_)				, ::boost::int8_t name_ ## X_
#define FIELD_TINYINT_UNSIGNED(name_)		, ::boost::uint8_t name_ ## X_
#define FIELD_SMALLINT(name_)				, ::boost::int16_t name_ ## X_
#define FIELD_SMALLINT_UNSIGNED(name_)		, ::boost::uint16_t name_ ## X_
#define FIELD_INTEGER(name_)				, ::boost::int32_t name_ ## X_
#define FIELD_INTEGER_UNSIGNED(name_)		, ::boost::uint32_t name_ ## X_
#define FIELD_BIGINT(name_)					, ::boost::int64_t name_ ## X_
#define FIELD_BIGINT_UNSIGNED(name_)		, ::boost::uint64_t name_ ## X_
#define FIELD_STRING(name_)					, ::std::string name_ ## X_
#define FIELD_DATETIME(name_)				, ::boost::uint64_t name_ ## X_

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
#undef FIELD_STRING
#undef FIELD_DATETIME

#define FIELD_BOOLEAN(name_)				, name_(name_ ## X_)
#define FIELD_TINYINT(name_)				, name_(name_ ## X_)
#define FIELD_TINYINT_UNSIGNED(name_)		, name_(name_ ## X_)
#define FIELD_SMALLINT(name_)				, name_(name_ ## X_)
#define FIELD_SMALLINT_UNSIGNED(name_)		, name_(name_ ## X_)
#define FIELD_INTEGER(name_)				, name_(name_ ## X_)
#define FIELD_INTEGER_UNSIGNED(name_)		, name_(name_ ## X_)
#define FIELD_BIGINT(name_)					, name_(name_ ## X_)
#define FIELD_BIGINT_UNSIGNED(name_)		, name_(name_ ## X_)
#define FIELD_STRING(name_)					, name_(STD_MOVE(name_ ## X_))
#define FIELD_DATETIME(name_)				, name_(name_ ## X_)

		MYSQL_OBJECT_FIELDS
	{
		::Poseidon::atomicFence(::Poseidon::ATOMIC_RELEASE);
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
#undef FIELD_STRING
#undef FIELD_DATETIME

#define FIELD_BOOLEAN(name_)	\
	bool get_ ## name_() const {	\
		return ::Poseidon::atomicLoad(name_, ::Poseidon::ATOMIC_CONSUME);	\
	}	\
	void set_ ## name_(bool val_, bool invalidates_ = true){	\
		::Poseidon::atomicStore(name_, val_, ::Poseidon::ATOMIC_RELEASE);	\
		if(invalidates_){	\
			invalidate();	\
		}	\
	}

#define FIELD_TINYINT(name_)	\
	::boost::int8_t get_ ## name_() const {	\
		return ::Poseidon::atomicLoad(name_, ::Poseidon::ATOMIC_CONSUME);	\
	}	\
	void set_ ## name_(::boost::int8_t val_, bool invalidates_ = true){	\
		::Poseidon::atomicStore(name_, val_, ::Poseidon::ATOMIC_RELEASE);	\
		if(invalidates_){	\
			invalidate();	\
		}	\
	}

#define FIELD_TINYINT_UNSIGNED(name_)	\
	::boost::uint8_t get_ ## name_() const {	\
		return ::Poseidon::atomicLoad(name_, ::Poseidon::ATOMIC_CONSUME);	\
	}	\
	void set_ ## name_(::boost::uint8_t val_, bool invalidates_ = true){	\
		::Poseidon::atomicStore(name_, val_, ::Poseidon::ATOMIC_RELEASE);	\
		if(invalidates_){	\
			invalidate();	\
		}	\
	}

#define FIELD_SMALLINT(name_)	\
	::boost::int16_t get_ ## name_() const {	\
		return ::Poseidon::atomicLoad(name_, ::Poseidon::ATOMIC_CONSUME);	\
	}	\
	void set_ ## name_(::boost::int16_t val_, bool invalidates_ = true){	\
		::Poseidon::atomicStore(name_, val_, ::Poseidon::ATOMIC_RELEASE);	\
		if(invalidates_){	\
			invalidate();	\
		}	\
	}

#define FIELD_SMALLINT_UNSIGNED(name_)	\
	::boost::uint16_t get_ ## name_() const {	\
		return ::Poseidon::atomicLoad(name_, ::Poseidon::ATOMIC_CONSUME);	\
	}	\
	void set_ ## name_(::boost::uint16_t val_, bool invalidates_ = true){	\
		::Poseidon::atomicStore(name_, val_, ::Poseidon::ATOMIC_RELEASE);	\
		if(invalidates_){	\
			invalidate();	\
		}	\
	}

#define FIELD_INTEGER(name_)	\
	::boost::int32_t get_ ## name_() const {	\
		return ::Poseidon::atomicLoad(name_, ::Poseidon::ATOMIC_CONSUME);	\
	}	\
	void set_ ## name_(::boost::int32_t val_, bool invalidates_ = true){	\
		::Poseidon::atomicStore(name_, val_, ::Poseidon::ATOMIC_RELEASE);	\
		if(invalidates_){	\
			invalidate();	\
		}	\
	}

#define FIELD_INTEGER_UNSIGNED(name_)	\
	::boost::uint32_t get_ ## name_() const {	\
		return ::Poseidon::atomicLoad(name_, ::Poseidon::ATOMIC_CONSUME);	\
	}	\
	void set_ ## name_(::boost::uint32_t val_, bool invalidates_ = true){	\
		::Poseidon::atomicStore(name_, val_, ::Poseidon::ATOMIC_RELEASE);	\
		if(invalidates_){	\
			invalidate();	\
		}	\
	}

#define FIELD_BIGINT(name_)	\
	::boost::int64_t get_ ## name_() const {	\
		return ::Poseidon::atomicLoad(name_, ::Poseidon::ATOMIC_CONSUME);	\
	}	\
	void set_ ## name_(::boost::int64_t val_, bool invalidates_ = true){	\
		::Poseidon::atomicStore(name_, val_, ::Poseidon::ATOMIC_RELEASE);	\
		if(invalidates_){	\
			invalidate();	\
		}	\
	}

#define FIELD_BIGINT_UNSIGNED(name_)	\
	::boost::uint64_t get_ ## name_() const {	\
		return ::Poseidon::atomicLoad(name_, ::Poseidon::ATOMIC_CONSUME);	\
	}	\
	void set_ ## name_(::boost::uint64_t val_, bool invalidates_ = true){	\
		::Poseidon::atomicStore(name_, val_, ::Poseidon::ATOMIC_RELEASE);	\
		if(invalidates_){	\
			invalidate();	\
		}	\
	}

#define FIELD_STRING(name_)	\
	const ::std::string & unlockedGet_ ## name_() const {	\
		return name_;	\
	}	\
	::std::string get_ ## name_() const {	\
		const ::Poseidon::Mutex::UniqueLock lock_(m_mutex);	\
		return name_;	\
	}	\
	void set_ ## name_(std::string val_, bool invalidates_ = true){	\
		{	\
			const ::Poseidon::Mutex::UniqueLock lock_(m_mutex);	\
			name_.swap(val_);	\
		}	\
		if(invalidates_){	\
			invalidate();	\
		}	\
	}

#define FIELD_DATETIME(name_)	\
	::boost::uint64_t get_ ## name_() const {	\
		return ::Poseidon::atomicLoad(name_, ::Poseidon::ATOMIC_CONSUME);	\
	}	\
	void set_ ## name_(::boost::uint64_t val_, bool invalidates_ = true){	\
		::Poseidon::atomicStore(name_, val_, ::Poseidon::ATOMIC_RELEASE);	\
		if(invalidates_){	\
			invalidate();	\
		}	\
	}

	MYSQL_OBJECT_FIELDS

	const char *getTableName() const {
		return TOKEN_TO_STR(MYSQL_OBJECT_NAME);
	}

	void syncGenerateSql(::std::string &sql_, bool toReplace_) const {
		::std::ostringstream oss_;

#undef FIELD_BOOLEAN
#undef FIELD_TINYINT
#undef FIELD_TINYINT_UNSIGNED
#undef FIELD_SMALLINT
#undef FIELD_SMALLINT_UNSIGNED
#undef FIELD_INTEGER
#undef FIELD_INTEGER_UNSIGNED
#undef FIELD_BIGINT
#undef FIELD_BIGINT_UNSIGNED
#undef FIELD_STRING
#undef FIELD_DATETIME

#define FIELD_BOOLEAN(name_)				(void)(oss_ <<", "),	\
												(void)(oss_ <<"`" TOKEN_TO_STR(name_) "` = "	\
													<<static_cast<long>(get_ ## name_())),
#define FIELD_TINYINT(name_)				(void)(oss_ <<", "),	\
												(void)(oss_ <<"`" TOKEN_TO_STR(name_) "` = "	\
													<<static_cast<long>(get_ ## name_())),
#define FIELD_TINYINT_UNSIGNED(name_)		(void)(oss_ <<", "),	\
												(void)(oss_ <<"`" TOKEN_TO_STR(name_) "` = "	\
													<<static_cast<unsigned long>(get_ ## name_())),
#define FIELD_SMALLINT(name_)				(void)(oss_ <<", "),	\
												(void)(oss_ <<"`" TOKEN_TO_STR(name_) "` = "	\
													<<static_cast<long>(get_ ## name_())),
#define FIELD_SMALLINT_UNSIGNED(name_)		(void)(oss_ <<", "),	\
												(void)(oss_ <<"`" TOKEN_TO_STR(name_) "` = "	\
													<<static_cast<unsigned long>(get_ ## name_())),
#define FIELD_INTEGER(name_)				(void)(oss_ <<", "),	\
												(void)(oss_ <<"`" TOKEN_TO_STR(name_) "` = "	\
													<<static_cast<long>(get_ ## name_())),
#define FIELD_INTEGER_UNSIGNED(name_)		(void)(oss_ <<", "),	\
												(void)(oss_ <<"`" TOKEN_TO_STR(name_) "` = "	\
													<<static_cast<unsigned long>(get_ ## name_())),
#define FIELD_BIGINT(name_)					(void)(oss_ <<", "),	\
												(void)(oss_ <<"`" TOKEN_TO_STR(name_) "` = "	\
													<<static_cast< ::boost::int64_t>(get_ ## name_())),
#define FIELD_BIGINT_UNSIGNED(name_)		(void)(oss_ <<", "),	\
												(void)(oss_ <<"`" TOKEN_TO_STR(name_) "` = "	\
													<<static_cast< ::boost::uint64_t>(get_ ## name_())),
#define FIELD_STRING(name_)					(void)(oss_ <<", "),	\
												(void)(oss_ <<"`" TOKEN_TO_STR(name_) "` = "	\
													<< '\''	\
													<< ::Poseidon::MySql::StringEscaper(get_ ## name_())	\
													<< '\''),
#define FIELD_DATETIME(name_)				(void)(oss_ <<", "),	\
												(void)(oss_ <<"`" TOKEN_TO_STR(name_) "` = "	\
													<< '\''	\
													<< ::Poseidon::MySql::DateFormatter(get_ ## name_())	\
													<< '\''),

		if(toReplace_){
			oss_ <<"REPLACE INTO `" TOKEN_TO_STR(MYSQL_OBJECT_NAME) "` SET ";
		} else {
			oss_ <<"INSERT INTO `" TOKEN_TO_STR(MYSQL_OBJECT_NAME) "` SET ";
		}
		STRIP_FIRST(MYSQL_OBJECT_FIELDS) (void)0;
		oss_.str().swap(sql_);
	}
	void syncFetch(const ::Poseidon::MySql::Connection &conn_){

#undef FIELD_BOOLEAN
#undef FIELD_TINYINT
#undef FIELD_TINYINT_UNSIGNED
#undef FIELD_SMALLINT
#undef FIELD_SMALLINT_UNSIGNED
#undef FIELD_INTEGER
#undef FIELD_INTEGER_UNSIGNED
#undef FIELD_BIGINT
#undef FIELD_BIGINT_UNSIGNED
#undef FIELD_STRING
#undef FIELD_DATETIME

#define FIELD_BOOLEAN(name_)				set_ ## name_(conn_.getSigned( TOKEN_TO_STR(name_) ));
#define FIELD_TINYINT(name_)				set_ ## name_(conn_.getSigned( TOKEN_TO_STR(name_) ));
#define FIELD_TINYINT_UNSIGNED(name_)		set_ ## name_(conn_.getUnsigned( TOKEN_TO_STR(name_) ));
#define FIELD_SMALLINT(name_)				set_ ## name_(conn_.getSigned( TOKEN_TO_STR(name_) ));
#define FIELD_SMALLINT_UNSIGNED(name_)		set_ ## name_(conn_.getUnsigned( TOKEN_TO_STR(name_) ));
#define FIELD_INTEGER(name_)				set_ ## name_(conn_.getSigned( TOKEN_TO_STR(name_) ));
#define FIELD_INTEGER_UNSIGNED(name_)		set_ ## name_(conn_.getUnsigned( TOKEN_TO_STR(name_) ));
#define FIELD_BIGINT(name_)					set_ ## name_(conn_.getSigned( TOKEN_TO_STR(name_) ));
#define FIELD_BIGINT_UNSIGNED(name_)		set_ ## name_(conn_.getUnsigned( TOKEN_TO_STR(name_) ));
#define FIELD_STRING(name_)					set_ ## name_(conn_.getString( TOKEN_TO_STR(name_) ));
#define FIELD_DATETIME(name_)				set_ ## name_(conn_.getDateTime( TOKEN_TO_STR(name_) ));

		MYSQL_OBJECT_FIELDS
	}
};

#undef MYSQL_OBJECT_NAME
#undef MYSQL_OBJECT_FIELDS
