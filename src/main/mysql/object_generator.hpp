#ifndef MYSQL_OBJECT_NAME
#	error MYSQL_OBJECT_NAME is undefined.
#endif

#ifndef MYSQL_OBJECT_FIELDS
#	error MYSQL_OBJECT_FIELDS is undefined.
#endif

#ifndef POSEIDON_MYSQL_OBJECT_BASE_HPP_
#	error Please #include "object_base.hpp" first.
#endif

#include "../cxx_ver.hpp"
#include "../cxx_util.hpp"
#include <sstream>
#include <cstdio>
#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>
#include <cppconn/connection.h>
#include <cppconn/prepared_statement.h>
#include <cppconn/resultset.h>
#include "../atomic.hpp"
#include "../log.hpp"

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

#define FIELD_BOOLEAN(name_)				, bool name_ = false
#define FIELD_TINYINT(name_)				, signed char name_ ## _ = 0
#define FIELD_TINYINT_UNSIGNED(name_)		, unsigned char name_ ## _ = 0
#define FIELD_SMALLINT(name_)				, short name_ ## _ = 0
#define FIELD_SMALLINT_UNSIGNED(name_)		, unsigned short name_ ## _ = 0
#define FIELD_INTEGER(name_)				, int name_ ## _ = 0
#define FIELD_INTEGER_UNSIGNED(name_)		, unsigned name_ ## _ = 0
#define FIELD_BIGINT(name_)					, long long name_ ## _ = 0
#define FIELD_BIGINT_UNSIGNED(name_)		, unsigned long long name_ ## _ = 0
#define FIELD_STRING(name_)					, ::std::string name_ ## _ = ::std::string()

	explicit MYSQL_OBJECT_NAME(STRIP_FIRST(void MYSQL_OBJECT_FIELDS))
		: MySqlObjectBase()

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

#define FIELD_BOOLEAN(name_)				, name_(name_ ## _)
#define FIELD_TINYINT(name_)				, name_(name_ ## _)
#define FIELD_TINYINT_UNSIGNED(name_)		, name_(name_ ## _)
#define FIELD_SMALLINT(name_)				, name_(name_ ## _)
#define FIELD_SMALLINT_UNSIGNED(name_)		, name_(name_ ## _)
#define FIELD_INTEGER(name_)				, name_(name_ ## _)
#define FIELD_INTEGER_UNSIGNED(name_)		, name_(name_ ## _)
#define FIELD_BIGINT(name_)					, name_(name_ ## _)
#define FIELD_BIGINT_UNSIGNED(name_)		, name_(name_ ## _)
#define FIELD_STRING(name_)					, name_(STD_MOVE(name_ ## _))

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
		return atomicLoad(name_);	\
	}	\
	void set_ ## name_(bool val_){	\
		atomicStore(name_, val_);	\
		invalidate();	\
	}

#define FIELD_TINYINT(name_)	\
	signed char get_ ## name_() const {	\
		return atomicLoad(name_);	\
	}	\
	void set_ ## name_(signed char val_){	\
		atomicStore(name_, val_);	\
		invalidate();	\
	}

#define FIELD_TINYINT_UNSIGNED(name_)	\
	unsigned char get_ ## name_() const {	\
		return atomicLoad(name_);	\
	}	\
	void set_ ## name_(unsigned char val_){	\
		atomicStore(name_, val_);	\
		invalidate();	\
	}

#define FIELD_SMALLINT(name_)	\
	short get_ ## name_() const {	\
		return atomicLoad(name_);	\
	}	\
	void set_ ## name_(short val_){	\
		atomicStore(name_, val_);	\
		invalidate();	\
	}

#define FIELD_SMALLINT_UNSIGNED(name_)	\
	unsigned short get_ ## name_() const {	\
		return atomicLoad(name_);	\
	}	\
	void set_ ## name_(unsigned short val_){	\
		atomicStore(name_, val_);	\
		invalidate();	\
	}

#define FIELD_INTEGER(name_)	\
	int get_ ## name_() const {	\
		return atomicLoad(name_);	\
	}	\
	void set_ ## name_(int val_){	\
		atomicStore(name_, val_);	\
		invalidate();	\
	}

#define FIELD_INTEGER_UNSIGNED(name_)	\
	unsigned get_ ## name_() const {	\
		return atomicLoad(name_);	\
	}	\
	void set_ ## name_(unsigned val_){	\
		atomicStore(name_, val_);	\
		invalidate();	\
	}

#define FIELD_BIGINT(name_)	\
	long long get_ ## name_() const {	\
		return atomicLoad(name_);	\
	}	\
	void set_ ## name_(long long val_){	\
		atomicStore(name_, val_);	\
		invalidate();	\
	}

#define FIELD_BIGINT_UNSIGNED(name_)	\
	unsigned long long get_ ## name_() const {	\
		return atomicLoad(name_);	\
	}	\
	void set_ ## name_(unsigned long long val_){	\
		atomicStore(name_, val_);	\
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

private:
	static void doQuery(boost::scoped_ptr<sql::ResultSet> &rs_,
		sql::Connection *conn_, const char *filter_, const char *limit_)
	{

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

#undef MYSQL_OBJECT_NAME_COMMA_
#define MYSQL_OBJECT_NAME_COMMA_(name_)	\
	"`" TOKEN_TO_STR(MYSQL_OBJECT_NAME) "`.`" TOKEN_TO_STR(name_) "`, "

#define FIELD_BOOLEAN(name_)				MYSQL_OBJECT_NAME_COMMA_(name_)
#define FIELD_TINYINT(name_)				MYSQL_OBJECT_NAME_COMMA_(name_)
#define FIELD_TINYINT_UNSIGNED(name_)		MYSQL_OBJECT_NAME_COMMA_(name_)
#define FIELD_SMALLINT(name_)				MYSQL_OBJECT_NAME_COMMA_(name_)
#define FIELD_SMALLINT_UNSIGNED(name_)		MYSQL_OBJECT_NAME_COMMA_(name_)
#define FIELD_INTEGER(name_)				MYSQL_OBJECT_NAME_COMMA_(name_)
#define FIELD_INTEGER_UNSIGNED(name_)		MYSQL_OBJECT_NAME_COMMA_(name_)
#define FIELD_BIGINT(name_)					MYSQL_OBJECT_NAME_COMMA_(name_)
#define FIELD_BIGINT_UNSIGNED(name_)		MYSQL_OBJECT_NAME_COMMA_(name_)
#define FIELD_STRING(name_)					MYSQL_OBJECT_NAME_COMMA_(name_)

		sql::SQLString str_("SELECT " MYSQL_OBJECT_FIELDS);
		str_->erase(str_->end() - 2, str_->end());
		str_->append(" FROM `" TOKEN_TO_STR(MYSQL_OBJECT_NAME) "` ");
		str_->append(filter_);
		str_->append(limit_);

		LOG_POSEIDON_DEBUG("Executing SQL in " TOKEN_TO_STR(MYSQL_OBJECT_NAME) ": ", str_);

		rs_.reset(conn_->prepareStatement(str_)->executeQuery());
	}
	void doFetch(sql::ResultSet *rs_){
		unsigned index_ = 0;

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

#define FIELD_BOOLEAN(name_)				name_ = rs_->getBoolean(++index_);
#define FIELD_TINYINT(name_)				name_ = rs_->getInt(++index_);
#define FIELD_TINYINT_UNSIGNED(name_)		name_ = rs_->getUInt(++index_);
#define FIELD_SMALLINT(name_)				name_ = rs_->getInt(++index_);
#define FIELD_SMALLINT_UNSIGNED(name_)		name_ = rs_->getUInt(++index_);
#define FIELD_INTEGER(name_)				name_ = rs_->getInt(++index_);
#define FIELD_INTEGER_UNSIGNED(name_)		name_ = rs_->getUInt(++index_);
#define FIELD_BIGINT(name_)					name_ = rs_->getInt64(++index_);
#define FIELD_BIGINT_UNSIGNED(name_)		name_ = rs_->getUInt64(++index_);
#define FIELD_STRING(name_)					rs_->getString(++index_)->swap(name_);

		MYSQL_OBJECT_FIELDS

		atomicSynchronize();
	}

private:
	void syncSave(sql::Connection *conn_) const {

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

#undef MYSQL_OBJECT_NAME_ASSIGN_COMMA_
#define MYSQL_OBJECT_NAME_ASSIGN_COMMA_(name_)	\
	"`" TOKEN_TO_STR(name_) "` = ?, "

#define FIELD_BOOLEAN(name_)				MYSQL_OBJECT_NAME_ASSIGN_COMMA_(name_)
#define FIELD_TINYINT(name_)				MYSQL_OBJECT_NAME_ASSIGN_COMMA_(name_)
#define FIELD_TINYINT_UNSIGNED(name_)		MYSQL_OBJECT_NAME_ASSIGN_COMMA_(name_)
#define FIELD_SMALLINT(name_)				MYSQL_OBJECT_NAME_ASSIGN_COMMA_(name_)
#define FIELD_SMALLINT_UNSIGNED(name_)		MYSQL_OBJECT_NAME_ASSIGN_COMMA_(name_)
#define FIELD_INTEGER(name_)				MYSQL_OBJECT_NAME_ASSIGN_COMMA_(name_)
#define FIELD_INTEGER_UNSIGNED(name_)		MYSQL_OBJECT_NAME_ASSIGN_COMMA_(name_)
#define FIELD_BIGINT(name_)					MYSQL_OBJECT_NAME_ASSIGN_COMMA_(name_)
#define FIELD_BIGINT_UNSIGNED(name_)		MYSQL_OBJECT_NAME_ASSIGN_COMMA_(name_)
#define FIELD_STRING(name_)					MYSQL_OBJECT_NAME_ASSIGN_COMMA_(name_)

		sql::SQLString str_(
			"REPLACE INTO `" TOKEN_TO_STR(MYSQL_OBJECT_NAME) "` "
			"SET " MYSQL_OBJECT_FIELDS);
		str_->erase(str_->end() - 2, str_->end());

		LOG_POSEIDON_DEBUG("Executing SQL in " TOKEN_TO_STR(MYSQL_OBJECT_NAME) ": ", str_);

		const boost::scoped_ptr<sql::PreparedStatement> ps_(conn_->prepareStatement(str_));
		std::vector<boost::shared_ptr<void> > contexts_;
		unsigned index_ = 0;

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

#define FIELD_BOOLEAN(name_)				ps_->setBoolean(++index_, get_ ## name_());
#define FIELD_TINYINT(name_)				ps_->setInt(++index_, get_ ## name_());
#define FIELD_TINYINT_UNSIGNED(name_)		ps_->setUInt(++index_, get_ ## name_());
#define FIELD_SMALLINT(name_)				ps_->setInt(++index_, get_ ## name_());
#define FIELD_SMALLINT_UNSIGNED(name_)		ps_->setUInt(++index_, get_ ## name_());
#define FIELD_INTEGER(name_)				ps_->setInt(++index_, get_ ## name_());
#define FIELD_INTEGER_UNSIGNED(name_)		ps_->setUInt(++index_, get_ ## name_());
#define FIELD_BIGINT(name_)					ps_->setInt64(++index_, get_ ## name_());
#define FIELD_BIGINT_UNSIGNED(name_)		ps_->setUInt64(++index_, get_ ## name_());
#define FIELD_STRING(name_)					if(get_ ## name_().size() <= 255){	\
												ps_->setString(++index_, get_ ## name_());	\
											} else {	\
												AUTO(ss_, ::boost::make_shared<	\
													::std::stringstream>(get_ ## name_()));	\
												contexts_.push_back(ss_);	\
												ps_->setBlob(++index_, ss_.get());	\
											}

		MYSQL_OBJECT_FIELDS

		ps_->executeUpdate();
	}
	bool syncLoad(sql::Connection *conn_, const char *filter_){
		boost::scoped_ptr<sql::ResultSet> rs_;
		doQuery(rs_, conn_, filter_, " LIMIT 1");
		if(!rs_->first()){
			return false;
		}
		doFetch(rs_.get());
		return true;
	}

public:
	static std::vector<boost::shared_ptr<MYSQL_OBJECT_NAME> >
		batchQuery(sql::Connection *conn_, const char *filter_)
	{
		std::vector<boost::shared_ptr<MYSQL_OBJECT_NAME> > ret_;
		boost::scoped_ptr<sql::ResultSet> rs_;
		doQuery(rs_, conn_, filter_, "");
		rs_->beforeFirst();
		while(rs_->next()){
			AUTO(obj_, boost::make_shared<MYSQL_OBJECT_NAME>());
			obj_->doFetch(rs_.get());
			ret_.push_back(STD_MOVE(obj_));
		}
		return ret_;
	}
};

#undef MYSQL_OBJECT_NAME
#undef MYSQL_OBJECT_FIELDS
