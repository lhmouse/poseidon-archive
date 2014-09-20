#ifndef MYSQL_OBJECT_NAME
#	error MYSQL_OBJECT_NAME is undefined.
#endif

#ifndef MYSQL_OBJECT_FIELDS
#	error MYSQL_OBJECT_FIELDS is undefined.
#endif

#ifndef POSEIDON_MYSQL_OBJECT_GENERATOR_HPP_
#define POSEIDON_MYSQL_OBJECT_GENERATOR_HPP_

#include "../../cxx_ver.hpp"
#include <sstream>
#include <cstdio>
#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>
#include <cppconn/connection.h>
#include <cppconn/prepared_statement.h>
#include <cppconn/resultset.h>
#include "object_base.hpp"
#include "../log.hpp"

#define MYSQL_OBJECT_TO_STR_2_(x_)			# x_
#define MYSQL_OBJECT_TO_STR_(x_)			MYSQL_OBJECT_TO_STR_2_(x_)

#endif

#ifdef MYSQL_OBJECT_NAMESPACE
namespace MYSQL_OBJECT_NAMESPACE {
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

#define FIELD_BOOLEAN(name_)				bool name_;
#define FIELD_TINYINT(name_)				signed char name_;
#define FIELD_TINYINT_UNSIGNED(name_)		unsigned char name_;
#define FIELD_SMALLINT(name_)				short name_;
#define FIELD_SMALLINT_UNSIGNED(name_)		unsigned short name_;
#define FIELD_INTEGER(name_)				int name_;
#define FIELD_INTEGER_UNSIGNED(name_)		unsigned name_;
#define FIELD_BIGINT(name_)					long long name_;
#define FIELD_BIGINT_UNSIGNED(name_)		unsigned long long name_;
#define FIELD_STRING(name_)					::std::string name_;

	MYSQL_OBJECT_FIELDS

public:
	MYSQL_OBJECT_NAME() throw()
		: MySqlObjectBase(MYSQL_OBJECT_TO_STR_(MYSQL_OBJECT_NAME))

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

#define FIELD_BOOLEAN(name_)				bool get_ ## name_() const { return name_; }	\
											void set_ ## name_(bool val_){ name_ = val_; asyncSave(); }
#define FIELD_TINYINT(name_)				signed char get_ ## name_() const { return name_; }	\
											void set_ ## name_(signed char val_){ name_ = val_; asyncSave(); }
#define FIELD_TINYINT_UNSIGNED(name_)		unsigned char get_ ## name_() const { return name_; }	\
											void set_ ## name_(unsigned char val_){ name_ = val_; asyncSave(); }
#define FIELD_SMALLINT(name_)				short get_ ## name_() const { return name_; }	\
											void set_ ## name_(short val_){ name_ = val_; asyncSave(); }
#define FIELD_SMALLINT_UNSIGNED(name_)		unsigned short get_ ## name_() const { return name_; }	\
											void set_ ## name_(unsigned short val_){ name_ = val_; asyncSave(); }
#define FIELD_INTEGER(name_)				int get_ ## name_() const { return name_; }	\
											void set_ ## name_(int val_){ name_ = val_; asyncSave(); }
#define FIELD_INTEGER_UNSIGNED(name_)		unsigned get_ ## name_() const { return name_; }	\
											void set_ ## name_(unsigned val_){ name_ = val_; asyncSave(); }
#define FIELD_BIGINT(name_)					long long get_ ## name_() const { return name_; }	\
											void set_ ## name_(long long val_){ name_ = val_; asyncSave(); }
#define FIELD_BIGINT_UNSIGNED(name_)		unsigned long long get_ ## name_() const { return name_; }	\
											void set_ ## name_(unsigned long long val_){ name_ = val_; asyncSave(); }
#define FIELD_STRING(name_)					const std::string & get_ ## name_() const { return name_; }	\
											void set_ ## name_(std::string val_){ name_.swap(val_); asyncSave(); }

	MYSQL_OBJECT_FIELDS

// ###TEMP
// private:
	void prepare(boost::scoped_ptr<sql::PreparedStatement> &ps_, sql::Connection *conn_){

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

#define FIELD_BOOLEAN(name_)				"`" MYSQL_OBJECT_TO_STR_(name_) "` = ?, "
#define FIELD_TINYINT(name_)				"`" MYSQL_OBJECT_TO_STR_(name_) "` = ?, "
#define FIELD_TINYINT_UNSIGNED(name_)		"`" MYSQL_OBJECT_TO_STR_(name_) "` = ?, "
#define FIELD_SMALLINT(name_)				"`" MYSQL_OBJECT_TO_STR_(name_) "` = ?, "
#define FIELD_SMALLINT_UNSIGNED(name_)		"`" MYSQL_OBJECT_TO_STR_(name_) "` = ?, "
#define FIELD_INTEGER(name_)				"`" MYSQL_OBJECT_TO_STR_(name_) "` = ?, "
#define FIELD_INTEGER_UNSIGNED(name_)		"`" MYSQL_OBJECT_TO_STR_(name_) "` = ?, "
#define FIELD_BIGINT(name_)					"`" MYSQL_OBJECT_TO_STR_(name_) "` = ?, "
#define FIELD_BIGINT_UNSIGNED(name_)		"`" MYSQL_OBJECT_TO_STR_(name_) "` = ?, "
#define FIELD_STRING(name_)					"`" MYSQL_OBJECT_TO_STR_(name_) "` = ?, "

		sql::SQLString str_(
			"REPLACE INTO `" MYSQL_OBJECT_TO_STR_(MYSQL_OBJECT_NAME) "` "
			"SET " MYSQL_OBJECT_FIELDS);
		str_->erase(str_->end() - 2, str_->end());

		LOG_DEBUG(MYSQL_OBJECT_TO_STR_(MYSQL_OBJECT_NAMESPACE), "::",
			MYSQL_OBJECT_TO_STR_(MYSQL_OBJECT_NAME), "::prepare(): ", str_);

		ps_.reset(conn_->prepareStatement(str_));
	}
	void pack(sql::PreparedStatement *ps_, std::vector<boost::any> &contexts_){
		(void)contexts_;

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

#define FIELD_BOOLEAN(name_)				ps_->setBoolean(++index_, name_);
#define FIELD_TINYINT(name_)				ps_->setInt(++index_, name_);
#define FIELD_TINYINT_UNSIGNED(name_)		ps_->setUInt(++index_, name_);
#define FIELD_SMALLINT(name_)				ps_->setInt(++index_, name_);
#define FIELD_SMALLINT_UNSIGNED(name_)		ps_->setUInt(++index_, name_);
#define FIELD_INTEGER(name_)				ps_->setInt(++index_, name_);
#define FIELD_INTEGER_UNSIGNED(name_)		ps_->setUInt(++index_, name_);
#define FIELD_BIGINT(name_)					ps_->setInt64(++index_, name_);
#define FIELD_BIGINT_UNSIGNED(name_)		ps_->setUInt64(++index_, name_);
#define FIELD_STRING(name_)					if(name_.size() <= 255){	\
												ps_->setString(++index_, name_);	\
											} else {	\
												::boost::shared_ptr< ::std::stringstream> ss_ =	\
													::boost::make_shared< ::std::stringstream>(name_);	\
												contexts_.push_back(ss_);	\
												ps_->setBlob(++index_, ss_.get());	\
											}

		MYSQL_OBJECT_FIELDS
	}

	static void doQuery(boost::scoped_ptr<sql::ResultSet> &rs_, sql::Connection *conn_,
		const char *filter_, const char *limit_)
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

#define FIELD_BOOLEAN(name_)				"`" MYSQL_OBJECT_TO_STR_(MYSQL_OBJECT_NAME) "`."	\
											"`" MYSQL_OBJECT_TO_STR_(name_) "`, "
#define FIELD_TINYINT(name_)				"`" MYSQL_OBJECT_TO_STR_(MYSQL_OBJECT_NAME) "`."	\
											"`" MYSQL_OBJECT_TO_STR_(name_) "`, "
#define FIELD_TINYINT_UNSIGNED(name_)		"`" MYSQL_OBJECT_TO_STR_(MYSQL_OBJECT_NAME) "`."	\
											"`" MYSQL_OBJECT_TO_STR_(name_) "`, "
#define FIELD_SMALLINT(name_)				"`" MYSQL_OBJECT_TO_STR_(MYSQL_OBJECT_NAME) "`."	\
											"`" MYSQL_OBJECT_TO_STR_(name_) "`, "
#define FIELD_SMALLINT_UNSIGNED(name_)		"`" MYSQL_OBJECT_TO_STR_(MYSQL_OBJECT_NAME) "`."	\
											"`" MYSQL_OBJECT_TO_STR_(name_) "`, "
#define FIELD_INTEGER(name_)				"`" MYSQL_OBJECT_TO_STR_(MYSQL_OBJECT_NAME) "`."	\
											"`" MYSQL_OBJECT_TO_STR_(name_) "`, "
#define FIELD_INTEGER_UNSIGNED(name_)		"`" MYSQL_OBJECT_TO_STR_(MYSQL_OBJECT_NAME) "`."	\
											"`" MYSQL_OBJECT_TO_STR_(name_) "`, "
#define FIELD_BIGINT(name_)					"`" MYSQL_OBJECT_TO_STR_(MYSQL_OBJECT_NAME) "`."	\
											"`" MYSQL_OBJECT_TO_STR_(name_) "`, "
#define FIELD_BIGINT_UNSIGNED(name_)		"`" MYSQL_OBJECT_TO_STR_(MYSQL_OBJECT_NAME) "`."	\
											"`" MYSQL_OBJECT_TO_STR_(name_) "`, "
#define FIELD_STRING(name_)					"`" MYSQL_OBJECT_TO_STR_(MYSQL_OBJECT_NAME) "`."	\
											"`" MYSQL_OBJECT_TO_STR_(name_) "`, "

		sql::SQLString str_("SELECT " MYSQL_OBJECT_FIELDS);
		str_->erase(str_->end() - 2, str_->end());
		str_->append(" FROM `" MYSQL_OBJECT_TO_STR_(MYSQL_OBJECT_NAME) "` ");
		str_->append(filter_);
		str_->append(limit_);

		LOG_DEBUG(MYSQL_OBJECT_TO_STR_(MYSQL_OBJECT_NAMESPACE), "::",
			MYSQL_OBJECT_TO_STR_(MYSQL_OBJECT_NAME), "::query(): ", str_);

		rs_.reset(conn_->prepareStatement(str_)->executeQuery());
	}

	bool query(boost::scoped_ptr<sql::ResultSet> &rs_, sql::Connection *conn_, const char *filter_) const {
		doQuery(rs_, conn_, filter_, " LIMIT 1");
		return rs_->first();
	}
	void fetch(sql::ResultSet *rs_){
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
			obj_->fetch(rs_.get());
			ret_.push_back(STD_MOVE(obj_));
		}
		return STD_MOVE(ret_);
	}
};

#ifdef MYSQL_OBJECT_NAMESPACE
}
#endif

#undef MYSQL_OBJECT_NAME
#undef MYSQL_OBJECT_FIELDS
