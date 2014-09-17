#ifndef DATABASE_TUPLE_NAME
#   error DATABASE_TUPLE_NAME is undefined.
#endif

#ifndef DATABASE_TUPLE_FIELDS
#   error DATABASE_TUPLE_FIELDS is undefined.
#endif

#ifndef POSEIDON_DATABASE_TUPLE_GENERATOR_HPP_
#define POSEIDON_DATABASE_TUPLE_GENERATOR_HPP_

#include "../../cxx_ver.hpp"
#include <string>
#include <sstream>
#include <vector>
#include <cstddef>
#include <boost/cstdint.hpp>
#include <boost/scoped_ptr.hpp>
#include <cppconn/prepared_statement.h>
#include <cppconn/resultset.h>
#include "../stream_buffer.hpp"

namespace Poseidon {

struct DatabaseTupleBase {
};

}

#endif // POSEIDON_DATABASE_TUPLE_GENERATOR_HPP_

#ifdef DATABASE_TUPLE_NAMESPACE
namespace DATABASE_TUPLE_NAMESPACE {
#endif

struct DATABASE_TUPLE_NAME
	: public ::Poseidon::DatabaseTupleBase
{

#undef FIELD_BOOLEAN
#undef FIELD_TINYINT
#undef FIELD_TINYINT_UNSIGNED
#undef FIELD_SMALLINT
#undef FIELD_SMALLINT_UNSIGNED
#undef FIELD_INT
#undef FIELD_INT_UNSIGNED
#undef FIELD_BIGINT
#undef FIELD_BIGINT_UNSIGNED
#undef FIELD_VARCHAR
#undef FIELD_BLOB

#define FIELD_BOOLEAN(name_)				bool name_;
#define FIELD_TINYINT(name_)				signed char name_;
#define FIELD_TINYINT_UNSIGNED(name_)		unsigned char name_;
#define FIELD_SMALLINT(name_)				short name_;
#define FIELD_SMALLINT_UNSIGNED(name_)		unsigned short name_;
#define FIELD_INT(name_)					int name_;
#define FIELD_INT_UNSIGNED(name_)			unsigned int name_;
#define FIELD_BIGINT(name_)					long long name_;
#define FIELD_BIGINT_UNSIGNED(name_)		unsigned long long name_;
#define FIELD_VARCHAR(name_)				::std::string name_;
#define FIELD_BLOB(name_)	\
	StreamBuffer name_;	\
	boost::scoped_ptr< ::std::stringstream> streamOf_ ## name_ ## _;

	DATABASE_TUPLE_FIELDS

	typedef std::vector<DATABASE_TUPLE_NAME> tuple_vector;

	DATABASE_TUPLE_NAME()
		: ::Poseidon::DatabaseTupleBase()

#undef FIELD_BOOLEAN
#undef FIELD_TINYINT
#undef FIELD_TINYINT_UNSIGNED
#undef FIELD_SMALLINT
#undef FIELD_SMALLINT_UNSIGNED
#undef FIELD_INT
#undef FIELD_INT_UNSIGNED
#undef FIELD_BIGINT
#undef FIELD_BIGINT_UNSIGNED
#undef FIELD_VARCHAR
#undef FIELD_BLOB

#define FIELD_BOOLEAN(name_)				, name_()
#define FIELD_TINYINT(name_)				, name_()
#define FIELD_TINYINT_UNSIGNED(name_)		, name_()
#define FIELD_SMALLINT(name_)				, name_()
#define FIELD_SMALLINT_UNSIGNED(name_)		, name_()
#define FIELD_INT(name_)					, name_()
#define FIELD_INT_UNSIGNED(name_)			, name_()
#define FIELD_BIGINT(name_)					, name_()
#define FIELD_BIGINT_UNSIGNED(name_)		, name_()
#define FIELD_VARCHAR(name_)				, name_()
#define FIELD_BLOB(name_)					, name_()

		DATABASE_TUPLE_FIELDS
	{
	}

	static void appendFieldNames(std::string &output_){
		(void)output_;

		if(fieldCount() == 0){
			return;
		}

#undef FIELD_BOOLEAN
#undef FIELD_TINYINT
#undef FIELD_TINYINT_UNSIGNED
#undef FIELD_SMALLINT
#undef FIELD_SMALLINT_UNSIGNED
#undef FIELD_INT
#undef FIELD_INT_UNSIGNED
#undef FIELD_BIGINT
#undef FIELD_BIGINT_UNSIGNED
#undef FIELD_VARCHAR
#undef FIELD_BLOB

#define FIELD_BOOLEAN(name_)				output_.append("`" #name_ "`, ");
#define FIELD_TINYINT(name_)				output_.append("`" #name_ "`, ");
#define FIELD_TINYINT_UNSIGNED(name_)		output_.append("`" #name_ "`, ");
#define FIELD_SMALLINT(name_)				output_.append("`" #name_ "`, ");
#define FIELD_SMALLINT_UNSIGNED(name_)		output_.append("`" #name_ "`, ");
#define FIELD_INT(name_)					output_.append("`" #name_ "`, ");
#define FIELD_INT_UNSIGNED(name_)			output_.append("`" #name_ "`, ");
#define FIELD_BIGINT(name_)					output_.append("`" #name_ "`, ");
#define FIELD_BIGINT_UNSIGNED(name_)		output_.append("`" #name_ "`, ");
#define FIELD_VARCHAR(name_)				output_.append("`" #name_ "`, ");
#define FIELD_BLOB(name_)					output_.append("`" #name_ "`, ");

		DATABASE_TUPLE_FIELDS

		output_.erase(output_.end() - 2);
	}

	static CONSTEXPR std::size_t fieldCount(){
		return 0

#undef FIELD_BOOLEAN
#undef FIELD_TINYINT
#undef FIELD_TINYINT_UNSIGNED
#undef FIELD_SMALLINT
#undef FIELD_SMALLINT_UNSIGNED
#undef FIELD_INT
#undef FIELD_INT_UNSIGNED
#undef FIELD_BIGINT
#undef FIELD_BIGINT_UNSIGNED
#undef FIELD_VARCHAR
#undef FIELD_BLOB

#define FIELD_BOOLEAN(name_)				+ 1
#define FIELD_TINYINT(name_)				+ 1
#define FIELD_TINYINT_UNSIGNED(name_)		+ 1
#define FIELD_SMALLINT(name_)				+ 1
#define FIELD_SMALLINT_UNSIGNED(name_)		+ 1
#define FIELD_INT(name_)					+ 1
#define FIELD_INT_UNSIGNED(name_)			+ 1
#define FIELD_BIGINT(name_)					+ 1
#define FIELD_BIGINT_UNSIGNED(name_)		+ 1
#define FIELD_VARCHAR(name_)				+ 1
#define FIELD_BLOB(name_)					+ 1

		DATABASE_TUPLE_FIELDS
		;
	}

	void serialize(::sql::PreparedStatement &ps_){
		(void)ps_;
		unsigned index_ = 0;
		(void)index_;

		std::stringstream *ss_;
		char buf_[1024];
		std::size_t count_;
		(void)ss_;
		(void)buf_;
		(void)count_;

#undef FIELD_BOOLEAN
#undef FIELD_TINYINT
#undef FIELD_TINYINT_UNSIGNED
#undef FIELD_SMALLINT
#undef FIELD_SMALLINT_UNSIGNED
#undef FIELD_INT
#undef FIELD_INT_UNSIGNED
#undef FIELD_BIGINT
#undef FIELD_BIGINT_UNSIGNED
#undef FIELD_VARCHAR
#undef FIELD_BLOB

#define FIELD_BOOLEAN(name_)				ps_.setBoolean(++index_, name_);
#define FIELD_TINYINT(name_)				ps_.setInt(++index_, name_);
#define FIELD_TINYINT_UNSIGNED(name_)		ps_.setUInt(++index_, name_);
#define FIELD_SMALLINT(name_)				ps_.setInt(++index_, name_);
#define FIELD_SMALLINT_UNSIGNED(name_)		ps_.setUInt(++index_, name_);
#define FIELD_INT(name_)					ps_.setInt(++index_, name_);
#define FIELD_INT_UNSIGNED(name_)			ps_.setUInt(++index_, name_);
#define FIELD_BIGINT(name_)					ps_.setInt64(++index_, name_);
#define FIELD_BIGINT_UNSIGNED(name_)		ps_.setUInt64(++index_, name_);
#define FIELD_VARCHAR(name_)				ps_.setString(++index_, name_);
#define FIELD_BLOB(name_)	\
		ss_ = new ::std::stringstream;	\
		(streamOf_ ## name_ ## _).reset(ss_);	\
		for(;;){	\
			count_ = name_.get(buf_, sizeof(buf_));	\
			if(count_ == 0){	\
				break;	\
			}	\
			ss_->write(buf_, count_);	\
		}

		DATABASE_TUPLE_FIELDS
	}

	void deserialize(const ::sql::ResultSet &rs_){
		(void)rs_;
		unsigned index_ = 0;
		(void)index_;

		std::istream *is_;
		char buf_[1024];
		std::size_t count_;
		(void)is_;
		(void)buf_;
		(void)count_;

#undef FIELD_BOOLEAN
#undef FIELD_TINYINT
#undef FIELD_TINYINT_UNSIGNED
#undef FIELD_SMALLINT
#undef FIELD_SMALLINT_UNSIGNED
#undef FIELD_INT
#undef FIELD_INT_UNSIGNED
#undef FIELD_BIGINT
#undef FIELD_BIGINT_UNSIGNED
#undef FIELD_VARCHAR
#undef FIELD_BLOB

#define FIELD_BOOLEAN(name_)				name_ = rs_.getBoolean(++index_);
#define FIELD_TINYINT(name_)				name_ = rs_.getInt(++index_);
#define FIELD_TINYINT_UNSIGNED(name_)		name_ = rs_.getUInt(++index_);
#define FIELD_SMALLINT(name_)				name_ = rs_.getInt(++index_);
#define FIELD_SMALLINT_UNSIGNED(name_)		name_ = rs_.getUInt(++index_);
#define FIELD_INT(name_)					name_ = rs_.getInt(++index_);
#define FIELD_INT_UNSIGNED(name_)			name_ = rs_.getUInt(++index_);
#define FIELD_BIGINT(name_)					name_ = rs_.getInt64(++index_);
#define FIELD_BIGINT_UNSIGNED(name_)		name_ = rs_.getUInt64(++index_);
#define FIELD_VARCHAR(name_)				name_ = rs_.getString(++index_);
#define FIELD_BLOB(name_)	\
		name_.clear();	\
		is_ = rs_.getBlob(++index_);	\
		for(;;){	\
			count_ = is_->readsome(buf_, sizeof(buf_));	\
			if(count_ == 0){	\
				break;	\
			}	\
			name_.put(buf_, count_);	\
		}

		DATABASE_TUPLE_FIELDS
	}
};

#ifdef DATABASE_TUPLE_NAMESPACE
}
#endif

#undef DATABASE_TUPLE_NAME
#undef DATABASE_TUPLE_FIELDS
