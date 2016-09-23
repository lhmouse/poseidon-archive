// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2016, LH_Mouse. All wrongs reserved.

#ifndef MONGODB_OBJECT_NAME
#   error MONGODB_OBJECT_NAME is undefined.
#endif

#ifndef MONGODB_OBJECT_FIELDS
#   error MONGODB_OBJECT_FIELDS is undefined.
#endif

#ifndef MONGODB_OBJECT_PRIMARY_KEY
#   error MONGODB_OBJECT_PRIMARY_KEY is undefined.
#endif

#ifndef POSEIDON_MONGODB_OBJECT_BASE_HPP_
#   error Please #include <poseidon/mongodb/object_base.hpp> first.
#endif

#pragma GCC push_options
#pragma GCC diagnostic ignored "-Wshadow"

class MONGODB_OBJECT_NAME : public ::Poseidon::MongoDb::ObjectBase {
public:
	static ::boost::shared_ptr< ::Poseidon::MongoDb::ObjectBase> create(){
		return ::boost::make_shared<MONGODB_OBJECT_NAME>();
	}

private:

#undef FIELD_BOOLEAN
#undef FIELD_SIGNED
#undef FIELD_UNSIGNED
#undef FIELD_DOUBLE
#undef FIELD_STRING
#undef FIELD_DATETIME
#undef FIELD_UUID
#undef FIELD_BLOB

#define FIELD_BOOLEAN(name_)                bool name_;
#define FIELD_SIGNED(name_)                 ::boost::int64_t name_;
#define FIELD_UNSIGNED(name_)               ::boost::uint64_t name_;
#define FIELD_DOUBLE(name_)                 double name_;
#define FIELD_STRING(name_)                 ::std::string name_;
#define FIELD_DATETIME(name_)               ::boost::uint64_t name_;
#define FIELD_UUID(name_)                   ::Poseidon::Uuid name_;
#define FIELD_BLOB(name_)                   ::std::string name_;

	MONGODB_OBJECT_FIELDS

public:
	MONGODB_OBJECT_NAME()
		: ::Poseidon::MongoDb::ObjectBase()

#undef FIELD_BOOLEAN
#undef FIELD_SIGNED
#undef FIELD_UNSIGNED
#undef FIELD_DOUBLE
#undef FIELD_STRING
#undef FIELD_DATETIME
#undef FIELD_UUID
#undef FIELD_BLOB

#define FIELD_BOOLEAN(name_)                , name_()
#define FIELD_SIGNED(name_)                 , name_()
#define FIELD_UNSIGNED(name_)               , name_()
#define FIELD_DOUBLE(name_)                 , name_()
#define FIELD_STRING(name_)                 , name_()
#define FIELD_DATETIME(name_)               , name_()
#define FIELD_UUID(name_)                   , name_()
#define FIELD_BLOB(name_)                   , name_()

		MONGODB_OBJECT_FIELDS
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

	explicit MONGODB_OBJECT_NAME(STRIP_FIRST(void MONGODB_OBJECT_FIELDS))
		: ::Poseidon::MongoDb::ObjectBase()

#undef FIELD_BOOLEAN
#undef FIELD_SIGNED
#undef FIELD_UNSIGNED
#undef FIELD_DOUBLE
#undef FIELD_STRING
#undef FIELD_DATETIME
#undef FIELD_UUID
#undef FIELD_BLOB

#define FIELD_BOOLEAN(name_)                , name_(name_ ## X_)
#define FIELD_SIGNED(name_)                 , name_(name_ ## X_)
#define FIELD_UNSIGNED(name_)               , name_(name_ ## X_)
#define FIELD_DOUBLE(name_)                 , name_(name_ ## X_)
#define FIELD_STRING(name_)                 , name_(STD_MOVE(name_ ## X_))
#define FIELD_DATETIME(name_)               , name_(name_ ## X_)
#define FIELD_UUID(name_)                   , name_(name_ ## X_)
#define FIELD_BLOB(name_)                   , name_(STD_MOVE(name_ ## X_))

		MONGODB_OBJECT_FIELDS
	{
		::Poseidon::atomic_fence(::Poseidon::ATOMIC_RELEASE);
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
#define FIELD_SIGNED(name_)                 const ::boost::int64_t & unlocked_get_ ## name_() const {	\
                                            	return name_;	\
                                            }	\
                                            ::boost::int64_t get_ ## name_() const {	\
                                            	return ::Poseidon::atomic_load(name_, ::Poseidon::ATOMIC_CONSUME);	\
                                            }	\
                                            void set_ ## name_(::boost::int64_t val_, bool invalidates_ = true){	\
                                            	::Poseidon::atomic_store(name_, val_, ::Poseidon::ATOMIC_RELEASE);	\
                                            	if(invalidates_){ invalidate(); }	\
                                            }
#define FIELD_UNSIGNED(name_)               const ::boost::uint64_t & unlocked_get_ ## name_() const {	\
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
#define FIELD_BLOB(name_)                   const ::std::string & unlocked_get_ ## name_() const {	\
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

	MONGODB_OBJECT_FIELDS

	const char *get_collection_name() const OVERRIDE {
		return TOKEN_TO_STR(MONGODB_OBJECT_NAME);
	}

	void generate_document(::Poseidon::MongoDb::BsonBuilder &doc) const OVERRIDE {

#undef FIELD_BOOLEAN
#undef FIELD_SIGNED
#undef FIELD_UNSIGNED
#undef FIELD_DOUBLE
#undef FIELD_STRING
#undef FIELD_DATETIME
#undef FIELD_UUID
#undef FIELD_BLOB

#define FIELD_BOOLEAN(name_)                doc.append_boolean  (::Poseidon::SharedNts::view(TOKEN_TO_STR(name_)), get_ ## name_());
#define FIELD_SIGNED(name_)                 doc.append_signed   (::Poseidon::SharedNts::view(TOKEN_TO_STR(name_)), get_ ## name_());
#define FIELD_UNSIGNED(name_)               doc.append_unsigned (::Poseidon::SharedNts::view(TOKEN_TO_STR(name_)), get_ ## name_());
#define FIELD_DOUBLE(name_)                 doc.append_double   (::Poseidon::SharedNts::view(TOKEN_TO_STR(name_)), get_ ## name_());
#define FIELD_STRING(name_)                 doc.append_string   (::Poseidon::SharedNts::view(TOKEN_TO_STR(name_)), get_ ## name_());
#define FIELD_DATETIME(name_)               doc.append_datetime (::Poseidon::SharedNts::view(TOKEN_TO_STR(name_)), get_ ## name_());
#define FIELD_UUID(name_)                   doc.append_uuid     (::Poseidon::SharedNts::view(TOKEN_TO_STR(name_)), get_ ## name_());
#define FIELD_BLOB(name_)                   doc.append_blob     (::Poseidon::SharedNts::view(TOKEN_TO_STR(name_)), get_ ## name_());

		MONGODB_OBJECT_FIELDS
	}
	void generate_primary_key(::Poseidon::MongoDb::BsonBuilder &query) const OVERRIDE {

#undef FIELD_BOOLEAN
#undef FIELD_SIGNED
#undef FIELD_UNSIGNED
#undef FIELD_DOUBLE
#undef FIELD_STRING
#undef FIELD_DATETIME
#undef FIELD_UUID
#undef FIELD_BLOB

#define FIELD_BOOLEAN(name_)                if(__builtin_strstr(TOKEN_TO_STR(MONGODB_OBJECT_PRIMARY_KEY), TOKEN_TO_STR(name_)))	\
                                            	{ query.append_boolean  (::Poseidon::SharedNts::view(TOKEN_TO_STR(name_)), get_ ## name_()); }
#define FIELD_SIGNED(name_)                 if(__builtin_strstr(TOKEN_TO_STR(MONGODB_OBJECT_PRIMARY_KEY), TOKEN_TO_STR(name_)))	\
                                            	{ query.append_signed   (::Poseidon::SharedNts::view(TOKEN_TO_STR(name_)), get_ ## name_()); }
#define FIELD_UNSIGNED(name_)               if(__builtin_strstr(TOKEN_TO_STR(MONGODB_OBJECT_PRIMARY_KEY), TOKEN_TO_STR(name_)))	\
                                            	{ query.append_unsigned (::Poseidon::SharedNts::view(TOKEN_TO_STR(name_)), get_ ## name_()); }
#define FIELD_DOUBLE(name_)                 if(__builtin_strstr(TOKEN_TO_STR(MONGODB_OBJECT_PRIMARY_KEY), TOKEN_TO_STR(name_)))	\
                                            	{ query.append_double   (::Poseidon::SharedNts::view(TOKEN_TO_STR(name_)), get_ ## name_()); }
#define FIELD_STRING(name_)                 if(__builtin_strstr(TOKEN_TO_STR(MONGODB_OBJECT_PRIMARY_KEY), TOKEN_TO_STR(name_)))	\
                                            	{ query.append_string   (::Poseidon::SharedNts::view(TOKEN_TO_STR(name_)), get_ ## name_()); }
#define FIELD_DATETIME(name_)               if(__builtin_strstr(TOKEN_TO_STR(MONGODB_OBJECT_PRIMARY_KEY), TOKEN_TO_STR(name_)))	\
                                            	{ query.append_datetime (::Poseidon::SharedNts::view(TOKEN_TO_STR(name_)), get_ ## name_()); }
#define FIELD_UUID(name_)                   if(__builtin_strstr(TOKEN_TO_STR(MONGODB_OBJECT_PRIMARY_KEY), TOKEN_TO_STR(name_)))	\
                                            	{ query.append_uuid     (::Poseidon::SharedNts::view(TOKEN_TO_STR(name_)), get_ ## name_()); }
#define FIELD_BLOB(name_)                   if(__builtin_strstr(TOKEN_TO_STR(MONGODB_OBJECT_PRIMARY_KEY), TOKEN_TO_STR(name_)))	\
                                            	{ query.append_blob     (::Poseidon::SharedNts::view(TOKEN_TO_STR(name_)), get_ ## name_()); }

		MONGODB_OBJECT_FIELDS
	}
	void fetch(const boost::shared_ptr<const ::Poseidon::MongoDb::Connection> &conn_) OVERRIDE {

#undef FIELD_BOOLEAN
#undef FIELD_SIGNED
#undef FIELD_UNSIGNED
#undef FIELD_DOUBLE
#undef FIELD_STRING
#undef FIELD_DATETIME
#undef FIELD_UUID
#undef FIELD_BLOB

#define FIELD_BOOLEAN(name_)                set_ ## name_(conn_->get_boolean  ( TOKEN_TO_STR(name_) ), false);
#define FIELD_SIGNED(name_)                 set_ ## name_(conn_->get_signed   ( TOKEN_TO_STR(name_) ), false);
#define FIELD_UNSIGNED(name_)               set_ ## name_(conn_->get_unsigned ( TOKEN_TO_STR(name_) ), false);
#define FIELD_DOUBLE(name_)                 set_ ## name_(conn_->get_double   ( TOKEN_TO_STR(name_) ), false);
#define FIELD_STRING(name_)                 set_ ## name_(conn_->get_string   ( TOKEN_TO_STR(name_) ), false);
#define FIELD_DATETIME(name_)               set_ ## name_(conn_->get_datetime ( TOKEN_TO_STR(name_) ), false);
#define FIELD_UUID(name_)                   set_ ## name_(conn_->get_uuid     ( TOKEN_TO_STR(name_) ), false);
#define FIELD_BLOB(name_)                   set_ ## name_(conn_->get_blob     ( TOKEN_TO_STR(name_) ), false);

		MONGODB_OBJECT_FIELDS
	}
};

#pragma GCC pop_options

#undef MONGODB_OBJECT_NAME
#undef MONGODB_OBJECT_FIELDS
#undef MONGODB_OBJECT_PRIMARY_KEY
