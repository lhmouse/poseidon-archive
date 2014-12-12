// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014, LH_Mouse. All wrongs reserved.

#ifndef MYSQL_OBJECT_NAME
#	error MYSQL_OBJECT_NAME is undefined.
#endif

#ifndef MYSQL_OBJECT_FIELDS
#	error MYSQL_OBJECT_FIELDS is undefined.
#endif

#ifndef POSEIDON_MYSQL_OBJECT_BASE_HPP_
#	error Please #include "object_base.hpp" first.
#endif

class MYSQL_OBJECT_NAME : public ::Poseidon::MySqlObjectBase {
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

#define FIELD_BOOLEAN(name_)				volatile bool name_;
#define FIELD_TINYINT(name_)				volatile signed char name_;
#define FIELD_TINYINT_UNSIGNED(name_)		volatile unsigned char name_;
#define FIELD_SMALLINT(name_)				volatile short name_;
#define FIELD_SMALLINT_UNSIGNED(name_)		volatile unsigned short name_;
#define FIELD_INTEGER(name_)				volatile int name_;
#define FIELD_INTEGER_UNSIGNED(name_)		volatile unsigned name_;
#define FIELD_BIGINT(name_)					volatile long long name_;
#define FIELD_BIGINT_UNSIGNED(name_)		volatile unsigned long long name_;
#define FIELD_STRING(name_)					::std::string name_;

	MYSQL_OBJECT_FIELDS

public:
	MYSQL_OBJECT_NAME()
		: ::Poseidon::MySqlObjectBase()

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

#define FIELD_BOOLEAN(name_)				, bool name_
#define FIELD_TINYINT(name_)				, ::boost::int8_t name_ ## X_
#define FIELD_TINYINT_UNSIGNED(name_)		, ::boost::uint8_t name_ ## X_
#define FIELD_SMALLINT(name_)				, ::boost::int16_t name_ ## X_
#define FIELD_SMALLINT_UNSIGNED(name_)		, ::boost::uint16_t name_ ## X_
#define FIELD_INTEGER(name_)				, ::boost::int32_t name_ ## X_
#define FIELD_INTEGER_UNSIGNED(name_)		, ::boost::uint32_t name_ ## X_
#define FIELD_BIGINT(name_)					, ::boost::int64_t name_ ## X_
#define FIELD_BIGINT_UNSIGNED(name_)		, ::boost::uint64_t name_ ## X_
#define FIELD_STRING(name_)					, ::std::string name_ ## X_

	explicit MYSQL_OBJECT_NAME(STRIP_FIRST(void MYSQL_OBJECT_FIELDS))
		: ::Poseidon::MySqlObjectBase()

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

		MYSQL_OBJECT_FIELDS
	{
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

#define FIELD_BOOLEAN(name_)	\
	bool get_ ## name_() const {	\
		return ::Poseidon::atomicLoad(name_, ATOMIC_ACQUIRE);	\
	}	\
	void set_ ## name_(bool val_){	\
		::Poseidon::atomicStore(name_, val_, ATOMIC_RELEASE);	\
		invalidate();	\
	}

#define FIELD_TINYINT(name_)	\
	signed char get_ ## name_() const {	\
		return ::Poseidon::atomicLoad(name_, ATOMIC_ACQUIRE);	\
	}	\
	void set_ ## name_(signed char val_){	\
		::Poseidon::atomicStore(name_, val_, ATOMIC_RELEASE);	\
		invalidate();	\
	}

#define FIELD_TINYINT_UNSIGNED(name_)	\
	unsigned char get_ ## name_() const {	\
		return ::Poseidon::atomicLoad(name_, ATOMIC_ACQUIRE);	\
	}	\
	void set_ ## name_(unsigned char val_){	\
		::Poseidon::atomicStore(name_, val_, ATOMIC_RELEASE);	\
		invalidate();	\
	}

#define FIELD_SMALLINT(name_)	\
	short get_ ## name_() const {	\
		return ::Poseidon::atomicLoad(name_, ATOMIC_ACQUIRE);	\
	}	\
	void set_ ## name_(short val_){	\
		::Poseidon::atomicStore(name_, val_, ATOMIC_RELEASE);	\
		invalidate();	\
	}

#define FIELD_SMALLINT_UNSIGNED(name_)	\
	unsigned short get_ ## name_() const {	\
		return ::Poseidon::atomicLoad(name_, ATOMIC_ACQUIRE);	\
	}	\
	void set_ ## name_(unsigned short val_){	\
		::Poseidon::atomicStore(name_, val_, ATOMIC_RELEASE);	\
		invalidate();	\
	}

#define FIELD_INTEGER(name_)	\
	int get_ ## name_() const {	\
		return ::Poseidon::atomicLoad(name_, ATOMIC_ACQUIRE);	\
	}	\
	void set_ ## name_(int val_){	\
		::Poseidon::atomicStore(name_, val_, ATOMIC_RELEASE);	\
		invalidate();	\
	}

#define FIELD_INTEGER_UNSIGNED(name_)	\
	unsigned get_ ## name_() const {	\
		return ::Poseidon::atomicLoad(name_, ATOMIC_ACQUIRE);	\
	}	\
	void set_ ## name_(unsigned val_){	\
		::Poseidon::atomicStore(name_, val_, ATOMIC_RELEASE);	\
		invalidate();	\
	}

#define FIELD_BIGINT(name_)	\
	long long get_ ## name_() const {	\
		return ::Poseidon::atomicLoad(name_, ATOMIC_ACQUIRE);	\
	}	\
	void set_ ## name_(long long val_){	\
		::Poseidon::atomicStore(name_, val_, ATOMIC_RELEASE);	\
		invalidate();	\
	}

#define FIELD_BIGINT_UNSIGNED(name_)	\
	unsigned long long get_ ## name_() const {	\
		return ::Poseidon::atomicLoad(name_, ATOMIC_ACQUIRE);	\
	}	\
	void set_ ## name_(unsigned long long val_){	\
		::Poseidon::atomicStore(name_, val_, ATOMIC_RELEASE);	\
		invalidate();	\
	}

#define FIELD_STRING(name_)	\
	::std::string get_ ## name_() const {	\
		const ::boost::shared_lock<boost::shared_mutex> slock(m_mutex);	\
		return name_;	\
	}	\
	void set_ ## name_(std::string val_){	\
		{	\
			const ::boost::unique_lock<boost::shared_mutex> ulock(m_mutex);	\
			name_.swap(val_);	\
		}	\
		invalidate();	\
	}

	MYSQL_OBJECT_FIELDS

public:
	const char *getTableName() const {
		return TOKEN_TO_STR(MYSQL_OBJECT_NAME);
	}

	void syncGenerateSql(::std::string &sql_, bool replaces_) const {
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

#define FIELD_BOOLEAN(name_)				(void)(oss_ <<", "),	\
												(void)(oss_ <<"`" TOKEN_TO_STR(name_) "` = "	\
													<<static_cast<unsigned long>(get_ ## name_())),
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
													<<static_cast<long long>(get_ ## name_())),
#define FIELD_BIGINT_UNSIGNED(name_)		(void)(oss_ <<", "),	\
												(void)(oss_ <<"`" TOKEN_TO_STR(name_) "` = "	\
													<<static_cast<unsigned long long>(get_ ## name_())),
#define FIELD_STRING(name_)					(void)(oss_ <<", "),	\
												(void)(oss_ <<"`" TOKEN_TO_STR(name_) "` = '"	\
													<< ::Poseidon::escapeStringForSql(get_ ## name_()) <<'\''),

		if(replaces_){
			oss_ <<"REPLACE INTO `" TOKEN_TO_STR(MYSQL_OBJECT_NAME) "` SET ";
		} else {
			oss_ <<"INSERT INTO `" TOKEN_TO_STR(MYSQL_OBJECT_NAME) "` SET ";
		}
		STRIP_FIRST(MYSQL_OBJECT_FIELDS) (void)0;
		oss_.str().swap(sql_);
	}
	void syncFetch(const ::Poseidon::MySqlConnection &conn_){

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

#define FIELD_BOOLEAN(name_)				set_ ## name_(conn_.getUnsigned( TOKEN_TO_STR(name_) ));
#define FIELD_TINYINT(name_)				set_ ## name_(conn_.getSigned( TOKEN_TO_STR(name_) ));
#define FIELD_TINYINT_UNSIGNED(name_)		set_ ## name_(conn_.getUnsigned( TOKEN_TO_STR(name_) ));
#define FIELD_SMALLINT(name_)				set_ ## name_(conn_.getSigned( TOKEN_TO_STR(name_) ));
#define FIELD_SMALLINT_UNSIGNED(name_)		set_ ## name_(conn_.getUnsigned( TOKEN_TO_STR(name_) ));
#define FIELD_INTEGER(name_)				set_ ## name_(conn_.getSigned( TOKEN_TO_STR(name_) ));
#define FIELD_INTEGER_UNSIGNED(name_)		set_ ## name_(conn_.getUnsigned( TOKEN_TO_STR(name_) ));
#define FIELD_BIGINT(name_)					set_ ## name_(conn_.getSigned( TOKEN_TO_STR(name_) ));
#define FIELD_BIGINT_UNSIGNED(name_)		set_ ## name_(conn_.getUnsigned( TOKEN_TO_STR(name_) ));
#define FIELD_STRING(name_)					set_ ## name_(conn_.getString( TOKEN_TO_STR(name_) ));

		MYSQL_OBJECT_FIELDS
	}

	static void batchAsyncLoad(std::string query, ::Poseidon::MySqlBatchAsyncLoadCallback callback){
		struct FactoryHelper {
			static boost::shared_ptr< ::Poseidon::MySqlObjectBase> create(){
				return boost::make_shared<MYSQL_OBJECT_NAME>();
			}
		};

		::Poseidon::MySqlObjectBase::batchAsyncLoad(TOKEN_TO_STR(MYSQL_OBJECT_NAME),
			STD_MOVE(query), &FactoryHelper::create, STD_MOVE(callback));
	}
};

#undef MYSQL_OBJECT_NAME
#undef MYSQL_OBJECT_FIELDS
