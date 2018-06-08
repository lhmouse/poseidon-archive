// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2018, LH_Mouse. All wrongs reserved.

#ifndef MESSAGE_NAME
#  error MESSAGE_NAME is undefined.
#endif

#ifndef MESSAGE_ID
#  error MESSAGE_ID is undefined.
#endif

#ifndef MESSAGE_FIELDS
#  error MESSAGE_FIELDS is undefined.
#endif

#ifndef POSEIDON_CBPP_MESSAGE_BASE_HPP_
#  error Please #include <poseidon/cbpp/message_base.hpp> first.
#endif

class MESSAGE_NAME : public ::Poseidon::Cbpp::Message_base {
public:
	enum { id = MESSAGE_ID };

public:

#undef FIELD_VINT
#undef FIELD_VUINT
#undef FIELD_FIXED
#undef FIELD_STRING
#undef FIELD_BLOB
#undef FIELD_FLEXIBLE
#undef FIELD_NESTED
#undef FIELD_ARRAY
#undef FIELD_LIST
#undef FIELD_REPEATED

#define FIELD_VINT(id_)             ::boost::int64_t id_;
#define FIELD_VUINT(id_)            ::boost::uint64_t id_;
#define FIELD_FIXED(id_, n_)        ::boost::array<unsigned char, n_> id_;
#define FIELD_STRING(id_)           ::std::string id_;
#define FIELD_BLOB(id_)             ::Poseidon::Stream_buffer id_;
#define FIELD_FLEXIBLE(id_)         ::Poseidon::Stream_buffer id_;
#define FIELD_NESTED(id_, Elem_)    Elem_ id_;
#define FIELD_ARRAY(id_, ...)       struct Unnamed_struct_##id_##_Stq_ { __VA_ARGS__ };	\
                                    ::boost::container::vector< Unnamed_struct_##id_##_Stq_ > id_;
#define FIELD_LIST(id_, ...)        struct Unnamed_struct_##id_##_Stq_ { __VA_ARGS__ };	\
                                    ::boost::container::deque< Unnamed_struct_##id_##_Stq_ > id_;
#define FIELD_REPEATED(id_, Elem_)  ::boost::container::deque< Elem_ > id_;

	MESSAGE_FIELDS

public:
	MESSAGE_NAME();
	~MESSAGE_NAME() OVERRIDE;

#undef FIELD_VINT
#undef FIELD_VUINT
#undef FIELD_FIXED
#undef FIELD_STRING
#undef FIELD_BLOB
#undef FIELD_FLEXIBLE
#undef FIELD_NESTED
#undef FIELD_ARRAY
#undef FIELD_LIST
#undef FIELD_REPEATED

#define FIELD_VINT(id_)             + 1
#define FIELD_VUINT(id_)            + 1
#define FIELD_FIXED(id_, n_)        + 1
#define FIELD_STRING(id_)           + 1
#define FIELD_BLOB(id_)             + 1
#define FIELD_FLEXIBLE(id_)         + 1
#define FIELD_NESTED(id_, Elem_)    + 1
#define FIELD_ARRAY(id_, ...)       + 1
#define FIELD_LIST(id_, ...)        + 1
#define FIELD_REPEATED(id_, Elem_)  + 1

#if (0 MESSAGE_FIELDS)

#undef FIELD_VINT
#undef FIELD_VUINT
#undef FIELD_FIXED
#undef FIELD_STRING
#undef FIELD_BLOB
#undef FIELD_FLEXIBLE
#undef FIELD_NESTED
#undef FIELD_ARRAY
#undef FIELD_LIST
#undef FIELD_REPEATED

#define FIELD_VINT(id_)             , ::boost::int64_t param_##id_##_Svy_
#define FIELD_VUINT(id_)            , ::boost::uint64_t param_##id_##_Svy_
#define FIELD_FIXED(id_, n_)        , const ::boost::array<unsigned char, n_> & param_##id_##_Svy_
#define FIELD_STRING(id_)           , ::std::string param_##id_##_Svy_
#define FIELD_BLOB(id_)             , ::Poseidon::Stream_buffer param_##id_##_Svy_
#define FIELD_FLEXIBLE(id_)         , ::Poseidon::Stream_buffer param_##id_##_Svy_
#define FIELD_NESTED(id_, Elem_)    , Elem_ param_##id_##_Svy_
#define FIELD_ARRAY(id_, ...)       , ::std::size_t param_##id_##_Svy_
#define FIELD_LIST(id_, ...)        , ::std::size_t param_##id_##_Svy_
#define FIELD_REPEATED(id_, Elem_)  , ::std::size_t param_##id_##_Svy_

	explicit MESSAGE_NAME(POSEIDON_LAZY(POSEIDON_REST, void MESSAGE_FIELDS));

#endif // (0 MESSAGE_FIELDS)

public:
	::boost::uint64_t get_id() const OVERRIDE;
	void serialize(::Poseidon::Stream_buffer &buffer_) const OVERRIDE;
	void deserialize(::Poseidon::Stream_buffer &buffer_) OVERRIDE;
	void dump_debug(::std::ostream &os_, int indent_initial_ = 0) const OVERRIDE;
};

#ifdef CBPP_MESSAGE_EMIT_EXTERNAL_DEFINITIONS
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wshadow"

MESSAGE_NAME::MESSAGE_NAME()
	: ::Poseidon::Cbpp::Message_base()

#undef FIELD_VINT
#undef FIELD_VUINT
#undef FIELD_FIXED
#undef FIELD_STRING
#undef FIELD_BLOB
#undef FIELD_FLEXIBLE
#undef FIELD_NESTED
#undef FIELD_ARRAY
#undef FIELD_LIST
#undef FIELD_REPEATED

#define FIELD_VINT(id_)             , id_()
#define FIELD_VUINT(id_)            , id_()
#define FIELD_FIXED(id_, n_)        , id_()
#define FIELD_STRING(id_)           , id_()
#define FIELD_BLOB(id_)             , id_()
#define FIELD_FLEXIBLE(id_)         , id_()
#define FIELD_NESTED(id_, Elem_)    , id_()
#define FIELD_ARRAY(id_, ...)       , id_()
#define FIELD_LIST(id_, ...)        , id_()
#define FIELD_REPEATED(id_, Elem_)  , id_()

	MESSAGE_FIELDS
{
	//
}

MESSAGE_NAME::~MESSAGE_NAME(){
	//
}

#undef FIELD_VINT
#undef FIELD_VUINT
#undef FIELD_FIXED
#undef FIELD_STRING
#undef FIELD_BLOB
#undef FIELD_FLEXIBLE
#undef FIELD_NESTED
#undef FIELD_ARRAY
#undef FIELD_LIST
#undef FIELD_REPEATED

#define FIELD_VINT(id_)             + 1
#define FIELD_VUINT(id_)            + 1
#define FIELD_FIXED(id_, n_)        + 1
#define FIELD_STRING(id_)           + 1
#define FIELD_BLOB(id_)             + 1
#define FIELD_FLEXIBLE(id_)         + 1
#define FIELD_NESTED(id_, Elem_)    + 1
#define FIELD_ARRAY(id_, ...)       + 1
#define FIELD_LIST(id_, ...)        + 1
#define FIELD_REPEATED(id_, Elem_)  + 1

#if (0 MESSAGE_FIELDS)

#undef FIELD_VINT
#undef FIELD_VUINT
#undef FIELD_FIXED
#undef FIELD_STRING
#undef FIELD_BLOB
#undef FIELD_FLEXIBLE
#undef FIELD_NESTED
#undef FIELD_ARRAY
#undef FIELD_LIST
#undef FIELD_REPEATED

#define FIELD_VINT(id_)             , ::boost::int64_t param_##id_##_Svy_
#define FIELD_VUINT(id_)            , ::boost::uint64_t param_##id_##_Svy_
#define FIELD_FIXED(id_, n_)        , const ::boost::array<unsigned char, n_> & param_##id_##_Svy_
#define FIELD_STRING(id_)           , ::std::string param_##id_##_Svy_
#define FIELD_BLOB(id_)             , ::Poseidon::Stream_buffer param_##id_##_Svy_
#define FIELD_FLEXIBLE(id_)         , ::Poseidon::Stream_buffer param_##id_##_Svy_
#define FIELD_NESTED(id_, Elem_)    , Elem_ param_##id_##_Svy_
#define FIELD_ARRAY(id_, ...)       , ::std::size_t param_##id_##_Svy_
#define FIELD_LIST(id_, ...)        , ::std::size_t param_##id_##_Svy_
#define FIELD_REPEATED(id_, Elem_)  , ::std::size_t param_##id_##_Svy_

MESSAGE_NAME::MESSAGE_NAME(POSEIDON_LAZY(POSEIDON_REST, void MESSAGE_FIELDS))
	: ::Poseidon::Cbpp::Message_base()

#undef FIELD_VINT
#undef FIELD_VUINT
#undef FIELD_FIXED
#undef FIELD_STRING
#undef FIELD_BLOB
#undef FIELD_FLEXIBLE
#undef FIELD_NESTED
#undef FIELD_ARRAY
#undef FIELD_LIST
#undef FIELD_REPEATED

#define FIELD_VINT(id_)             , id_(param_##id_##_Svy_)
#define FIELD_VUINT(id_)            , id_(param_##id_##_Svy_)
#define FIELD_FIXED(id_, n_)        , id_(param_##id_##_Svy_)
#define FIELD_STRING(id_)           , id_(STD_MOVE(param_##id_##_Svy_))
#define FIELD_BLOB(id_)             , id_(STD_MOVE(param_##id_##_Svy_))
#define FIELD_FLEXIBLE(id_)         , id_(STD_MOVE(param_##id_##_Svy_))
#define FIELD_NESTED(id_, Elem_)    , id_(STD_MOVE(param_##id_##_Svy_))
#define FIELD_ARRAY(id_, ...)       , id_(param_##id_##_Svy_)
#define FIELD_LIST(id_, ...)        , id_(param_##id_##_Svy_)
#define FIELD_REPEATED(id_, Elem_)  , id_(param_##id_##_Svy_)

	MESSAGE_FIELDS
{
	//
}

#endif // (0 MESSAGE_FIELDS)

::boost::uint64_t MESSAGE_NAME::get_id() const {
	return MESSAGE_ID;
}
void MESSAGE_NAME::serialize(::Poseidon::Stream_buffer &buffer_) const {
	POSEIDON_PROFILE_ME;

	const AUTO(cur_, this);
	AUTO_REF(buf_, buffer_);

#undef FIELD_VINT
#undef FIELD_VUINT
#undef FIELD_FIXED
#undef FIELD_STRING
#undef FIELD_BLOB
#undef FIELD_FLEXIBLE
#undef FIELD_NESTED
#undef FIELD_ARRAY
#undef FIELD_LIST
#undef FIELD_REPEATED

#define FIELD_VINT(id_)             ::Poseidon::Cbpp::push_vint(buf_, cur_->id_);
#define FIELD_VUINT(id_)            ::Poseidon::Cbpp::push_vuint(buf_, cur_->id_);
#define FIELD_FIXED(id_, n_)        ::Poseidon::Cbpp::push_fixed(buf_, cur_->id_.data(), cur_->id_.size());
#define FIELD_STRING(id_)           ::Poseidon::Cbpp::push_string(buf_, cur_->id_);
#define FIELD_BLOB(id_)             ::Poseidon::Cbpp::push_blob(buf_, cur_->id_);
#define FIELD_FLEXIBLE(id_)         ::Poseidon::Cbpp::push_flexible(buf_, cur_->id_);
#define FIELD_NESTED(id_, Elem_)    cur_->id_.serialize(buf_);
#define FIELD_ARRAY(id_, ...)       {	\
                                      ::Poseidon::Cbpp::push_vuint(buf_, cur_->id_.size());	\
                                      for(AUTO(it_, cur_->id_.begin()); it_ != cur_->id_.end(); ++it_){	\
                                        {	\
                                          const AUTO(cur_, &*it_);	\
                                          __VA_ARGS__	\
                                        }	\
                                      }	\
                                    }
#define FIELD_LIST(id_, ...)        {	\
                                      ::Poseidon::Stream_buffer chunk_;	\
                                      for(AUTO(it_, cur_->id_.begin()); it_ != cur_->id_.end(); ++it_){	\
                                        {	\
                                          const AUTO(cur_, &*it_);	\
                                          AUTO_REF(buf_, chunk_);	\
                                          __VA_ARGS__	\
                                        }	\
                                        ::Poseidon::Cbpp::push_blob(buf_, chunk_);	\
                                        chunk_.clear();	\
                                      }	\
                                      buf_.put(0);	\
                                    }
#define FIELD_REPEATED(id_, Elem_)  {	\
                                      ::Poseidon::Stream_buffer chunk_;	\
                                      for(AUTO(it_, cur_->id_.begin()); it_ != cur_->id_.end(); ++it_){	\
                                        {	\
                                          const AUTO(cur_, &*it_);	\
                                          AUTO_REF(buf_, chunk_);	\
                                          cur_->serialize(buf_);	\
                                        }	\
                                        ::Poseidon::Cbpp::push_blob(buf_, chunk_);	\
                                        chunk_.clear();	\
                                      }	\
                                      buf_.put(0);	\
                                    }

	MESSAGE_FIELDS
}
void MESSAGE_NAME::deserialize(::Poseidon::Stream_buffer &buffer_){
	POSEIDON_PROFILE_ME;

	const AUTO(cur_, this);
	AUTO_REF(buf_, buffer_);

#undef FIELD_VINT
#undef FIELD_VUINT
#undef FIELD_FIXED
#undef FIELD_STRING
#undef FIELD_BLOB
#undef FIELD_FLEXIBLE
#undef FIELD_NESTED
#undef FIELD_ARRAY
#undef FIELD_LIST
#undef FIELD_REPEATED

#define FIELD_VINT(id_)             ::Poseidon::Cbpp::shift_vint(cur_->id_, buf_, POSEIDON_STRINGIFY(id_));
#define FIELD_VUINT(id_)            ::Poseidon::Cbpp::shift_vuint(cur_->id_, buf_, POSEIDON_STRINGIFY(id_));
#define FIELD_FIXED(id_, n_)        ::Poseidon::Cbpp::shift_fixed(cur_->id_.data(), cur_->id_.size(), buf_, POSEIDON_STRINGIFY(id_));
#define FIELD_STRING(id_)           ::Poseidon::Cbpp::shift_string(cur_->id_, buf_, POSEIDON_STRINGIFY(id_));
#define FIELD_BLOB(id_)             ::Poseidon::Cbpp::shift_blob(cur_->id_, buf_, POSEIDON_STRINGIFY(id_));
#define FIELD_FLEXIBLE(id_)         ::Poseidon::Cbpp::shift_flexible(cur_->id_, buf_, POSEIDON_STRINGIFY(id_));
#define FIELD_NESTED(id_, Elem_)    cur_->id_.deserialize(buf_);
#define FIELD_ARRAY(id_, ...)       {	\
                                      cur_->id_.clear();	\
                                      ::boost::uint64_t length_;	\
                                      ::Poseidon::Cbpp::shift_vuint(length_, buf_, POSEIDON_STRINGIFY(id_) ".length");	\
                                      for(;;){	\
                                        if(length_ == 0){	\
                                          break;	\
                                        }	\
                                        --length_;	\
                                        const AUTO(it_, cur_->id_.emplace(cur_->id_.end()));	\
                                        {	\
                                          const AUTO(cur_, &*it_);	\
                                          __VA_ARGS__	\
                                        }	\
                                      }	\
                                    }
#define FIELD_LIST(id_, ...)        {	\
                                      cur_->id_.clear();	\
                                      ::Poseidon::Stream_buffer chunk_;	\
                                      for(;;){	\
                                        ::Poseidon::Cbpp::shift_blob(chunk_, buf_, POSEIDON_STRINGIFY(id_) ".chunk");	\
                                        if(chunk_.empty()){	\
                                          break;	\
                                        }	\
                                        const AUTO(it_, cur_->id_.emplace(cur_->id_.end()));	\
                                        {	\
                                          const AUTO(cur_, &*it_);	\
                                          AUTO_REF(buf_, chunk_);	\
                                          __VA_ARGS__	\
                                        }	\
                                      }	\
                                    }
#define FIELD_REPEATED(id_, Elem_)  {	\
                                      cur_->id_.clear();	\
                                      ::Poseidon::Stream_buffer chunk_;	\
                                      for(;;){	\
                                        ::Poseidon::Cbpp::shift_blob(chunk_, buf_, POSEIDON_STRINGIFY(id_) ".chunk");	\
                                        if(chunk_.empty()){	\
                                          break;	\
                                        }	\
                                        const AUTO(it_, cur_->id_.emplace(cur_->id_.end()));	\
                                        {	\
                                          const AUTO(cur_, &*it_);	\
                                          AUTO_REF(buf_, chunk_);	\
                                          cur_->deserialize(buf_);	\
                                        }	\
                                      }	\
                                    }

	MESSAGE_FIELDS
}
void MESSAGE_NAME::dump_debug(::std::ostream &os_, int indent_initial_) const {
	POSEIDON_PROFILE_ME;

	static CONSTEXPR int s_indent_step_ = 2;
	const AUTO(cur_, this);
	int indent_ = indent_initial_;

#undef FIELD_VINT
#undef FIELD_VUINT
#undef FIELD_FIXED
#undef FIELD_STRING
#undef FIELD_BLOB
#undef FIELD_FLEXIBLE
#undef FIELD_NESTED
#undef FIELD_ARRAY
#undef FIELD_LIST
#undef FIELD_REPEATED

#define FIELD_VINT(id_)             {	\
                                      os_ << ::std::setw(indent_) <<"" <<POSEIDON_STRINGIFY(id_) <<": vint = " <<cur_->id_ << ::std::endl;	\
                                    }
#define FIELD_VUINT(id_)            {	\
                                      os_ << ::std::setw(indent_) <<"" <<POSEIDON_STRINGIFY(id_) <<": vuint = " <<cur_->id_ << ::std::endl;	\
                                    }
#define FIELD_FIXED(id_, n_)        {	\
                                      os_ << ::std::setw(indent_) <<"" <<POSEIDON_STRINGIFY(id_) <<": fixed(" <<cur_->id_.size() <<") = ";	\
                                      os_ << ::Poseidon::Hex_printer(cur_->id_.data(), cur_->id_.size());	\
                                      os_ << ::std::endl;	\
                                    }
#define FIELD_STRING(id_)           {	\
                                      os_ << ::std::setw(indent_) <<"" <<POSEIDON_STRINGIFY(id_) <<": string(" <<cur_->id_.size() <<") = " <<cur_->id_ << ::std::endl;	\
                                    }
#define FIELD_BLOB(id_)             {	\
                                      os_ << ::std::setw(indent_) <<"" <<POSEIDON_STRINGIFY(id_) <<": flexible(" <<cur_->id_.size() <<") = ";	\
                                      const void *data_;	\
                                      ::std::size_t size_;	\
                                      ::Poseidon::Stream_buffer::Enumeration_cookie cookie_;	\
                                      while(cur_->id_.enumerate_chunk(&data_, &size_, cookie_)){	\
                                        os_ << ::Poseidon::Hex_printer(data_, size_);	\
                                      }	\
                                      os_ << ::std::endl;	\
                                    }
#define FIELD_FLEXIBLE(id_)         {	\
                                      os_ << ::std::setw(indent_) <<"" <<POSEIDON_STRINGIFY(id_) <<": flexible(" <<cur_->id_.size() <<") = ";	\
                                      const void *data_;	\
                                      ::std::size_t size_;	\
                                      ::Poseidon::Stream_buffer::Enumeration_cookie cookie_;	\
                                      while(cur_->id_.enumerate_chunk(&data_, &size_, cookie_)){	\
                                        os_ << ::Poseidon::Hex_printer(data_, size_);	\
                                      }	\
                                      os_ << ::std::endl;	\
                                    }
#define FIELD_NESTED(id_, Elem_)    {	\
                                      cur_->id_.dump_debug(os_, indent_);	\
                                    }
#define FIELD_ARRAY(id_, ...)       {	\
                                      os_ << ::std::setw(indent_) <<"" <<POSEIDON_STRINGIFY(id_) <<": array(" <<cur_->id_.size() <<") = [" << ::std::endl;	\
                                      indent_ += s_indent_step_;	\
                                      for(AUTO(it_, cur_->id_.begin()); it_ != cur_->id_.end(); ++it_){	\
                                        os_ << ::std::setw(indent_) <<"" <<"{" << ::std::endl;	\
                                        indent_ += s_indent_step_;	\
                                        {	\
                                          const AUTO(cur_, &*it_);	\
                                          __VA_ARGS__	\
                                        }	\
                                        indent_ -= s_indent_step_;	\
                                        os_ << ::std::setw(indent_) <<"" <<"}" << ::std::endl;	\
                                      }	\
                                      indent_ -= s_indent_step_;	\
                                      os_ << ::std::setw(indent_) <<"" <<"]" << ::std::endl;	\
                                    }
#define FIELD_LIST(id_, ...)        {	\
                                      os_ << ::std::setw(indent_) <<"" <<POSEIDON_STRINGIFY(id_) <<": list(" <<cur_->id_.size() <<") = [" << ::std::endl;	\
                                      indent_ += s_indent_step_;	\
                                      for(AUTO(it_, cur_->id_.begin()); it_ != cur_->id_.end(); ++it_){	\
                                        os_ << ::std::setw(indent_) <<"" <<"{" << ::std::endl;	\
                                        indent_ += s_indent_step_;	\
                                        {	\
                                          const AUTO(cur_, &*it_);	\
                                          __VA_ARGS__	\
                                        }	\
                                        indent_ -= s_indent_step_;	\
                                        os_ << ::std::setw(indent_) <<"" <<"}" << ::std::endl;	\
                                      }	\
                                      indent_ -= s_indent_step_;	\
                                      os_ << ::std::setw(indent_) <<"" <<"]" << ::std::endl;	\
                                    }
#define FIELD_REPEATED(id_, Elem_)  {	\
                                      os_ << ::std::setw(indent_) <<"" <<POSEIDON_STRINGIFY(id_) <<": repeated(" <<cur_->id_.size() <<") = [" << ::std::endl;	\
                                      indent_ += s_indent_step_;	\
                                      for(AUTO(it_, cur_->id_.begin()); it_ != cur_->id_.end(); ++it_){	\
                                        const AUTO(cur_, &*it_);	\
                                        cur_->dump_debug(os_, indent_);	\
                                      }	\
                                      indent_ -= s_indent_step_;	\
                                      os_ << ::std::setw(indent_) <<"" <<"]" << ::std::endl;	\
                                    }

	os_ << std::setw(indent_) <<"" <<POSEIDON_LAZY(POSEIDON_STRINGIFY, MESSAGE_NAME) <<"(" <<get_id() <<") = {" << ::std::endl;
	indent_ += s_indent_step_;
	MESSAGE_FIELDS
	indent_ -= s_indent_step_;
	os_ << std::setw(indent_) <<"" <<"}" << ::std::endl;
}

#pragma GCC diagnostic pop
#endif // CBPP_MESSAGE_EMIT_EXTERNAL_DEFINITIONS

#undef MESSAGE_NAME
#undef MESSAGE_ID
#undef MESSAGE_FIELDS
