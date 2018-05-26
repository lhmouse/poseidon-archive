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
#undef FIELD_ARRAY
#undef FIELD_LIST
#undef FIELD_NESTED
#undef FIELD_REPEATED

#define FIELD_VINT(id_)             ::boost::int64_t id_;
#define FIELD_VUINT(id_)            ::boost::uint64_t id_;
#define FIELD_FIXED(id_, n_)        ::boost::array<unsigned char, n_> id_;
#define FIELD_STRING(id_)           ::std::string id_;
#define FIELD_BLOB(id_)             ::std::basic_string<unsigned char> id_;
#define FIELD_FLEXIBLE(id_)         ::Poseidon::Stream_buffer id_;
#define FIELD_ARRAY(id_, ...)       struct Cbpp##id_##F_ { __VA_ARGS__ };	\
                                    ::boost::container::vector< Cbpp##id_##F_ > id_;
#define FIELD_LIST(id_, ...)        struct Cbpp##id_##F_ { __VA_ARGS__ };	\
                                    ::boost::container::deque< Cbpp##id_##F_ > id_;
#define FIELD_NESTED(id_, Elem_)    Elem_ id_;
#define FIELD_REPEATED(id_, Elem_)  ::boost::container::deque< Elem_ > id_;

	MESSAGE_FIELDS

public:
	MESSAGE_NAME();

	explicit MESSAGE_NAME(::Poseidon::Stream_buffer buffer_);

	~MESSAGE_NAME() OVERRIDE;

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
#undef FIELD_ARRAY
#undef FIELD_LIST
#undef FIELD_NESTED
#undef FIELD_REPEATED

#define FIELD_VINT(id_)             , id_()
#define FIELD_VUINT(id_)            , id_()
#define FIELD_FIXED(id_, n_)        , id_()
#define FIELD_STRING(id_)           , id_()
#define FIELD_BLOB(id_)             , id_()
#define FIELD_FLEXIBLE(id_)         , id_()
#define FIELD_ARRAY(id_, ...)       , id_()
#define FIELD_LIST(id_, ...)        , id_()
#define FIELD_NESTED(id_, Elem_)    , id_()
#define FIELD_REPEATED(id_, Elem_)  , id_()

	MESSAGE_FIELDS
{
	//
}

MESSAGE_NAME::MESSAGE_NAME(::Poseidon::Stream_buffer buffer_)
	: ::Poseidon::Cbpp::Message_base()
{
	deserialize(buffer_);
	if(!buffer_.empty()){
		THROW_JUNK_AFTER_PACKET_(MESSAGE_NAME);
	}
}

MESSAGE_NAME::~MESSAGE_NAME(){
	//
}

::boost::uint64_t MESSAGE_NAME::get_id() const {
	return MESSAGE_ID;
}
void MESSAGE_NAME::serialize(::Poseidon::Stream_buffer &buffer_) const {
	const AUTO(cur_, this);
	AUTO_REF(buf_, buffer_);

#undef FIELD_VINT
#undef FIELD_VUINT
#undef FIELD_FIXED
#undef FIELD_STRING
#undef FIELD_BLOB
#undef FIELD_FLEXIBLE
#undef FIELD_ARRAY
#undef FIELD_LIST
#undef FIELD_NESTED
#undef FIELD_REPEATED

#define FIELD_VINT(id_)             {	\
                                      unsigned char temp_[16];	\
                                      unsigned char *wptr_ = temp_;	\
                                      ::Poseidon::vint64_to_binary(cur_->id_, wptr_);	\
                                      buf_.put(temp_, static_cast< ::std::size_t>(wptr_ - temp_));	\
                                    }
#define FIELD_VUINT(id_)            {	\
                                      unsigned char temp_[16];	\
                                      unsigned char *wptr_ = temp_;	\
                                      ::Poseidon::vuint64_to_binary(cur_->id_, wptr_);	\
                                      buf_.put(temp_, static_cast< ::std::size_t>(wptr_ - temp_));	\
                                    }
#define FIELD_FIXED(id_, n_)        {	\
                                      buf_.put(cur_->id_.data(), cur_->id_.size());	\
                                    }
#define FIELD_STRING(id_)           {	\
                                      unsigned char temp_[16];	\
                                      unsigned char *wptr_ = temp_;	\
                                      ::Poseidon::vuint64_to_binary(cur_->id_.size(), wptr_);	\
                                      buf_.put(temp_, static_cast< ::std::size_t>(wptr_ - temp_));	\
                                      buf_.put(cur_->id_);	\
                                    }
#define FIELD_BLOB(id_)             {	\
                                      unsigned char temp_[16];	\
                                      unsigned char *wptr_ = temp_;	\
                                      ::Poseidon::vuint64_to_binary(cur_->id_.size(), wptr_);	\
                                      buf_.put(temp_, static_cast< ::std::size_t>(wptr_ - temp_));	\
                                      buf_.put(cur_->id_);	\
                                    }
#define FIELD_FLEXIBLE(id_)         {	\
                                      buf_.put(cur_->id_);	\
                                    }
#define FIELD_ARRAY(id_, ...)       {	\
                                      unsigned char temp_[16];	\
                                      unsigned char *wptr_ = temp_;	\
                                      ::Poseidon::vuint64_to_binary(cur_->id_.size(), wptr_);	\
                                      buf_.put(temp_, static_cast< ::std::size_t>(wptr_ - temp_));	\
                                      for(AUTO(it_, cur_->id_.begin()); it_ != cur_->id_.end(); ++it_){	\
                                        const AUTO(cur_, &*it_);	\
                                        __VA_ARGS__	\
                                      }	\
                                    }
#define FIELD_LIST(id_, ...)        {	\
                                      for(AUTO(it_, cur_->id_.begin()); it_ != cur_->id_.end(); ++it_){	\
                                        ::Poseidon::Stream_buffer chunk_buf_;	\
                                        {	\
                                          const AUTO(cur_, &*it_);	\
                                          AUTO_REF(buf_, chunk_buf_);	\
                                          __VA_ARGS__	\
                                        }	\
                                        unsigned char temp_[16];	\
                                        unsigned char *wptr_ = temp_;	\
                                        ::Poseidon::vuint64_to_binary(chunk_buf_.size(), wptr_);	\
                                        buf_.put(temp_, static_cast< ::std::size_t>(wptr_ - temp_));	\
                                        buf_.splice(chunk_buf_);	\
                                      }	\
                                      buf_.put(0);	\
                                    }
#define FIELD_NESTED(id_, Elem_)    {	\
                                      cur_->id_.serialize(buf_);	\
                                    }
#define FIELD_REPEATED(id_, Elem_)  {	\
                                      for(AUTO(it_, cur_->id_.begin()); it_ != cur_->id_.end(); ++it_){	\
                                        ::Poseidon::Stream_buffer chunk_buf_;	\
                                        {	\
                                          const AUTO(cur_, &*it_);	\
                                          AUTO_REF(buf_, chunk_buf_);	\
                                          cur_->serialize(buf_);	\
                                        }	\
                                        unsigned char temp_[16];	\
                                        unsigned char *wptr_ = temp_;	\
                                        ::Poseidon::vuint64_to_binary(chunk_buf_.size(), wptr_);	\
                                        buf_.put(temp_, static_cast< ::std::size_t>(wptr_ - temp_));	\
                                        buf_.splice(chunk_buf_);	\
                                      }	\
                                      buf_.put(0);	\
                                    }

	MESSAGE_FIELDS
}
void MESSAGE_NAME::deserialize(::Poseidon::Stream_buffer &buffer_){
	const AUTO(cur_, this);
	AUTO_REF(buf_, buffer_);

#undef FIELD_VINT
#undef FIELD_VUINT
#undef FIELD_FIXED
#undef FIELD_STRING
#undef FIELD_BLOB
#undef FIELD_FLEXIBLE
#undef FIELD_ARRAY
#undef FIELD_LIST
#undef FIELD_NESTED
#undef FIELD_REPEATED

#define FIELD_VINT(id_)             {	\
                                      unsigned char temp_[16];	\
                                      unsigned char *rptr_ = temp_;	\
                                      const ::std::size_t nmax_ = buf_.peek(temp_, sizeof(temp_));	\
                                      if(!::Poseidon::vint64_from_binary(cur_->id_, rptr_, nmax_)){	\
                                        THROW_END_OF_STREAM_(MESSAGE_NAME, id_);	\
                                      }	\
                                      buf_.discard(static_cast< ::std::size_t>(rptr_ - temp_));	\
                                    }
#define FIELD_VUINT(id_)            {	\
                                      unsigned char temp_[16];	\
                                      unsigned char *rptr_ = temp_;	\
                                      const ::std::size_t nmax_ = buf_.peek(temp_, sizeof(temp_));	\
                                      if(!::Poseidon::vuint64_from_binary(cur_->id_, rptr_, nmax_)){	\
                                        THROW_END_OF_STREAM_(MESSAGE_NAME, id_);	\
                                      }	\
                                      buf_.discard(static_cast< ::std::size_t>(rptr_ - temp_));	\
                                    }
#define FIELD_FIXED(id_, n_)        {	\
                                      ::boost::uint64_t nreq_ = cur_->id_.size();	\
                                      ::std::size_t ngot_ = buf_.get(cur_->id_.data(), nreq_);	\
                                      if(ngot_ < nreq_){	\
                                        THROW_END_OF_STREAM_(MESSAGE_NAME, id_);	\
                                      }	\
                                    }
#define FIELD_STRING(id_)           {	\
                                      ::boost::uint64_t nreq_;	\
                                      unsigned char temp_[16];	\
                                      unsigned char *rptr_ = temp_;	\
                                      const ::std::size_t nmax_ = buf_.peek(temp_, sizeof(temp_));	\
                                      if(!::Poseidon::vuint64_from_binary(nreq_, rptr_, nmax_)){	\
                                        THROW_END_OF_STREAM_(MESSAGE_NAME, id_);	\
                                      }	\
                                      buf_.discard(static_cast< ::std::size_t>(rptr_ - temp_));	\
                                      if(nreq_ > SIZE_MAX){	\
                                        THROW_LENGTH_ERROR_(MESSAGE_NAME, id_);	\
                                      }	\
                                      cur_->id_.resize(nreq_);	\
                                      ::std::size_t ngot_ = buf_.get(&*cur_->id_.begin(), nreq_);	\
                                      if(ngot_ < nreq_){	\
                                        THROW_END_OF_STREAM_(MESSAGE_NAME, id_);	\
                                      }	\
                                    }
#define FIELD_BLOB(id_)             {	\
                                      ::boost::uint64_t nreq_;	\
                                      unsigned char temp_[16];	\
                                      unsigned char *rptr_ = temp_;	\
                                      const ::std::size_t nmax_ = buf_.peek(temp_, sizeof(temp_));	\
                                      if(!::Poseidon::vuint64_from_binary(nreq_, rptr_, nmax_)){	\
                                        THROW_END_OF_STREAM_(MESSAGE_NAME, id_);	\
                                      }	\
                                      buf_.discard(static_cast< ::std::size_t>(rptr_ - temp_));	\
                                      if(nreq_ > SIZE_MAX){	\
                                        THROW_LENGTH_ERROR_(MESSAGE_NAME, id_);	\
                                      }	\
                                      cur_->id_.resize(nreq_);	\
                                      ::std::size_t ngot_ = buf_.get(&*cur_->id_.begin(), nreq_);	\
                                      if(ngot_ < nreq_){	\
                                        THROW_END_OF_STREAM_(MESSAGE_NAME, id_);	\
                                      }	\
                                    }
#define FIELD_FLEXIBLE(id_)         {	\
                                      cur_->id_.clear();	\
                                      cur_->id_.swap(buf_);	\
                                    }
#define FIELD_ARRAY(id_, ...)       {	\
                                      cur_->id_.clear();	\
                                      ::boost::uint64_t nreq_;	\
                                      unsigned char temp_[16];	\
                                      unsigned char *rptr_ = temp_;	\
                                      const ::std::size_t nmax_ = buf_.peek(temp_, sizeof(temp_));	\
                                      if(!::Poseidon::vuint64_from_binary(nreq_, rptr_, nmax_)){	\
                                        THROW_END_OF_STREAM_(MESSAGE_NAME, id_);	\
                                      }	\
                                      buf_.discard(static_cast< ::std::size_t>(rptr_ - temp_));	\
                                      if(nreq_ > SIZE_MAX){	\
                                        THROW_LENGTH_ERROR_(MESSAGE_NAME, id_);	\
                                      }	\
                                      cur_->id_.reserve(nreq_);	\
                                      for(::std::size_t ncur_ = 0; ncur_ < nreq_; ++ncur_){	\
                                        const AUTO(it_, cur_->id_.emplace(cur_->id_.end()));	\
                                        const AUTO(cur_, &*it_);	\
                                        __VA_ARGS__	\
                                      }	\
                                    }
#define FIELD_LIST(id_, ...)        {	\
                                      cur_->id_.clear();	\
                                      for(;;){	\
                                        ::boost::uint64_t nreq_;	\
                                        unsigned char temp_[16];	\
                                        unsigned char *rptr_ = temp_;	\
                                        const ::std::size_t nmax_ = buf_.peek(temp_, sizeof(temp_));	\
                                        if(!::Poseidon::vuint64_from_binary(nreq_, rptr_, nmax_)){	\
                                          THROW_END_OF_STREAM_(MESSAGE_NAME, id_);	\
                                        }	\
                                        buf_.discard(static_cast< ::std::size_t>(rptr_ - temp_));	\
                                        if(nreq_ == 0){	\
                                          break;	\
                                        }	\
                                        if(nreq_ > SIZE_MAX){	\
                                          THROW_LENGTH_ERROR_(MESSAGE_NAME, id_);	\
                                        }	\
                                        ::Poseidon::Stream_buffer chunk_buf_ = buf_.cut_off(nreq_);	\
                                        if(chunk_buf_.size() < nreq_){	\
                                          THROW_END_OF_STREAM_(MESSAGE_NAME, id_);	\
                                        }	\
                                        {	\
                                          const AUTO(it_, cur_->id_.emplace(cur_->id_.end()));	\
                                          const AUTO(cur_, &*it_);	\
                                          AUTO_REF(buf_, chunk_buf_);	\
                                          __VA_ARGS__	\
                                        }	\
                                      }	\
                                    }
#define FIELD_NESTED(id_, Elem_)    {	\
                                      cur_->id_.deserialize(buf_);	\
                                    }
#define FIELD_REPEATED(id_, Elem_)  {	\
                                      cur_->id_.clear();	\
                                      for(;;){	\
                                        ::boost::uint64_t nreq_;	\
                                        unsigned char temp_[16];	\
                                        unsigned char *rptr_ = temp_;	\
                                        const ::std::size_t nmax_ = buf_.peek(temp_, sizeof(temp_));	\
                                        if(!::Poseidon::vuint64_from_binary(nreq_, rptr_, nmax_)){	\
                                          THROW_END_OF_STREAM_(MESSAGE_NAME, id_);	\
                                        }	\
                                        buf_.discard(static_cast< ::std::size_t>(rptr_ - temp_));	\
                                        if(nreq_ == 0){	\
                                          break;	\
                                        }	\
                                        if(nreq_ > SIZE_MAX){	\
                                          THROW_LENGTH_ERROR_(MESSAGE_NAME, id_);	\
                                        }	\
                                        ::Poseidon::Stream_buffer chunk_buf_ = buf_.cut_off(nreq_);	\
                                        if(chunk_buf_.size() < nreq_){	\
                                          THROW_END_OF_STREAM_(MESSAGE_NAME, id_);	\
                                        }	\
                                        {	\
                                          const AUTO(it_, cur_->id_.emplace(cur_->id_.end()));	\
                                          const AUTO(cur_, &*it_);	\
                                          AUTO_REF(buf_, chunk_buf_);	\
                                          cur_->deserialize(buf_);	\
                                        }	\
                                      }	\
                                    }

	MESSAGE_FIELDS
}
void MESSAGE_NAME::dump_debug(::std::ostream &os_, int indent_initial_) const {
	static CONSTEXPR int s_indent_step_ = 2;

	const AUTO(cur_, this);
	int indent_ = indent_initial_;

#undef FIELD_VINT
#undef FIELD_VUINT
#undef FIELD_FIXED
#undef FIELD_STRING
#undef FIELD_BLOB
#undef FIELD_FLEXIBLE
#undef FIELD_ARRAY
#undef FIELD_LIST
#undef FIELD_NESTED
#undef FIELD_REPEATED

#define FIELD_VINT(id_)             {	\
                                      os_ << ::std::setw(indent_) <<"" <<TOKEN_TO_STR(id_) <<": vint = " <<cur_->id_ << ::std::endl;	\
                                    }
#define FIELD_VUINT(id_)            {	\
                                      os_ << ::std::setw(indent_) <<"" <<TOKEN_TO_STR(id_) <<": vuint = " <<cur_->id_ << ::std::endl;	\
                                    }
#define FIELD_FIXED(id_, n_)        {	\
                                      os_ << ::std::setw(indent_) <<"" <<TOKEN_TO_STR(id_) <<": fixed(" <<cur_->id_.size() <<") = ";	\
                                      os_ << ::Poseidon::Hex_printer(cur_->id_.data(), cur_->id_.size());	\
                                      os_ << ::std::endl;	\
                                    }
#define FIELD_STRING(id_)           {	\
                                      os_ << ::std::setw(indent_) <<"" <<TOKEN_TO_STR(id_) <<": string(" <<cur_->id_.size() <<") = " <<cur_->id_ << ::std::endl;	\
                                    }
#define FIELD_BLOB(id_)             {	\
                                      os_ << ::std::setw(indent_) <<"" <<TOKEN_TO_STR(id_) <<": blob(" <<cur_->id_.size() <<") = ";	\
                                      os_ << ::Poseidon::Hex_printer(cur_->id_.data(), cur_->id_.size());	\
                                      os_ << ::std::endl;	\
                                    }
#define FIELD_FLEXIBLE(id_)         {	\
                                      os_ << ::std::setw(indent_) <<"" <<TOKEN_TO_STR(id_) <<": flexible(" <<cur_->id_.size() <<") = ";	\
                                      const void *data_;	\
                                      ::std::size_t size_;	\
                                      ::Poseidon::Stream_buffer::Enumeration_cookie cookie_;	\
                                      while(cur_->id_.enumerate_chunk(&data_, &size_, cookie_)){	\
                                        os_ << ::Poseidon::Hex_printer(data_, size_);	\
                                      }	\
                                      os_ << ::std::endl;	\
                                    }
#define FIELD_ARRAY(id_, ...)       {	\
                                      os_ << ::std::setw(indent_) <<"" <<TOKEN_TO_STR(id_) <<": array(" <<cur_->id_.size() <<") = [" << ::std::endl;	\
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
                                      os_ << ::std::setw(indent_) <<"" <<TOKEN_TO_STR(id_) <<": list(" <<cur_->id_.size() <<") = [" << ::std::endl;	\
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
#define FIELD_NESTED(id_, Elem_)    {	\
                                      cur_->id_.dump_debug(os_, indent_);	\
                                    }
#define FIELD_REPEATED(id_, Elem_)  {	\
                                      os_ << ::std::setw(indent_) <<"" <<TOKEN_TO_STR(id_) <<": nested(" <<cur_->id_.size() <<") = [" << ::std::endl;	\
                                      indent_ += s_indent_step_;	\
                                      for(AUTO(it_, cur_->id_.begin()); it_ != cur_->id_.end(); ++it_){	\
                                        const AUTO(cur_, &*it_);	\
                                        cur_->dump_debug(os_, indent_);	\
                                      }	\
                                      indent_ -= s_indent_step_;	\
                                      os_ << ::std::setw(indent_) <<"" <<"]" << ::std::endl;	\
                                    }

	os_ << std::setw(indent_) <<"" <<TOKEN_TO_STR(MESSAGE_NAME) <<"(" <<get_id() <<") = {" << ::std::endl;
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
