// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2017, LH_Mouse. All wrongs reserved.

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

class MESSAGE_NAME : public ::Poseidon::Cbpp::MessageBase {
public:
	enum {
		ID = MESSAGE_ID,
	};

public:

#undef FIELD_VINT
#undef FIELD_VUINT
#undef FIELD_FIXED
#undef FIELD_STRING
#undef FIELD_BLOB
#undef FIELD_FLEXIBLE
#undef FIELD_ARRAY
#undef FIELD_LIST

#define FIELD_VINT(id_)           ::boost::int64_t id_;
#define FIELD_VUINT(id_)          ::boost::uint64_t id_;
#define FIELD_FIXED(id_, n_)      ::boost::array<unsigned char, n_> id_;
#define FIELD_STRING(id_)         ::std::string id_;
#define FIELD_BLOB(id_)           ::std::basic_string<unsigned char> id_;
#define FIELD_FLEXIBLE(id_)       ::std::basic_string<unsigned char> id_;
#define FIELD_ARRAY(id_, ...)     struct Cbpp ## id_ ## F_ { __VA_ARGS__ };	\
                                  ::boost::container::vector< Cbpp ## id_ ## F_ > id_;
#define FIELD_LIST(id_, ...)      struct Cbpp ## id_ ## F_ { __VA_ARGS__ };	\
                                  ::boost::container::deque< Cbpp ## id_ ## F_ > id_;

	MESSAGE_FIELDS

public:

#undef FIELD_VINT
#undef FIELD_VUINT
#undef FIELD_FIXED
#undef FIELD_STRING
#undef FIELD_BLOB
#undef FIELD_FLEXIBLE
#undef FIELD_ARRAY
#undef FIELD_LIST

#define FIELD_VINT(id_)           + 1
#define FIELD_VUINT(id_)          + 1
#define FIELD_FIXED(id_, n_)      + 1
#define FIELD_STRING(id_)         + 1
#define FIELD_BLOB(id_)           + 1
#define FIELD_FLEXIBLE(id_)       + 1
#define FIELD_ARRAY(id_, ...)     + 1
#define FIELD_LIST(id_, ...)      + 1

#if (0 MESSAGE_FIELDS) != 0
	MESSAGE_NAME();
#endif // (0 MESSAGE_FIELDS) != 0

#undef FIELD_VINT
#undef FIELD_VUINT
#undef FIELD_FIXED
#undef FIELD_STRING
#undef FIELD_BLOB
#undef FIELD_FLEXIBLE
#undef FIELD_ARRAY
#undef FIELD_LIST

#define FIELD_VINT(id_)           , ::boost::int64_t id_ ## X_
#define FIELD_VUINT(id_)          , ::boost::uint64_t id_ ## X_
#define FIELD_FIXED(id_, n_)      , const ::boost::array<unsigned char, n_> & id_ ## X_
#define FIELD_STRING(id_)         , ::std::string id_ ## X_
#define FIELD_BLOB(id_)           , ::std::basic_string<unsigned char> id_ ## X_
#define FIELD_FLEXIBLE(id_)       , ::std::basic_string<unsigned char> id_ ## X_
#define FIELD_ARRAY(id_, ...)     , ::boost::container::vector< Cbpp ## id_ ## F_ > id_ ## X_
#define FIELD_LIST(id_, ...)      , ::boost::container::deque< Cbpp ## id_ ## F_ > id_ ## X_

	explicit MESSAGE_NAME(STRIP_FIRST(void MESSAGE_FIELDS));

	explicit MESSAGE_NAME(::Poseidon::StreamBuffer buffer_);

	~MESSAGE_NAME() OVERRIDE;

public:
	boost::uint64_t get_id() const OVERRIDE;
	void serialize(::Poseidon::StreamBuffer &buffer_) const OVERRIDE;
	void deserialize(::Poseidon::StreamBuffer &buffer_) OVERRIDE;
	void dump_debug(::std::ostream &os_) const OVERRIDE;
};

#ifdef CBPP_MESSAGE_EMIT_EXTERNAL_DEFINITIONS
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wshadow"

#undef FIELD_VINT
#undef FIELD_VUINT
#undef FIELD_FIXED
#undef FIELD_STRING
#undef FIELD_BLOB
#undef FIELD_FLEXIBLE
#undef FIELD_ARRAY
#undef FIELD_LIST

#define FIELD_VINT(id_)           + 1
#define FIELD_VUINT(id_)          + 1
#define FIELD_FIXED(id_, n_)      + 1
#define FIELD_STRING(id_)         + 1
#define FIELD_BLOB(id_)           + 1
#define FIELD_FLEXIBLE(id_)       + 1
#define FIELD_ARRAY(id_, ...)     + 1
#define FIELD_LIST(id_, ...)      + 1

#if (0 MESSAGE_FIELDS) != 0
MESSAGE_NAME::MESSAGE_NAME()
	: ::Poseidon::Cbpp::MessageBase()

#undef FIELD_VINT
#undef FIELD_VUINT
#undef FIELD_FIXED
#undef FIELD_STRING
#undef FIELD_BLOB
#undef FIELD_FLEXIBLE
#undef FIELD_ARRAY
#undef FIELD_LIST

#define FIELD_VINT(id_)           , id_()
#define FIELD_VUINT(id_)          , id_()
#define FIELD_FIXED(id_, n_)      , id_()
#define FIELD_STRING(id_)         , id_()
#define FIELD_BLOB(id_)           , id_()
#define FIELD_FLEXIBLE(id_)       , id_()
#define FIELD_ARRAY(id_, ...)     , id_()
#define FIELD_LIST(id_, ...)      , id_()

	MESSAGE_FIELDS
{ }
#endif // (0 MESSAGE_FIELDS) != 0

#undef FIELD_VINT
#undef FIELD_VUINT
#undef FIELD_FIXED
#undef FIELD_STRING
#undef FIELD_BLOB
#undef FIELD_FLEXIBLE
#undef FIELD_ARRAY
#undef FIELD_LIST

#define FIELD_VINT(id_)           , ::boost::int64_t id_ ## X_
#define FIELD_VUINT(id_)          , ::boost::uint64_t id_ ## X_
#define FIELD_FIXED(id_, n_)      , const ::boost::array<unsigned char, n_> & id_ ## X_
#define FIELD_STRING(id_)         , ::std::string id_ ## X_
#define FIELD_BLOB(id_)           , ::std::basic_string<unsigned char> id_ ## X_
#define FIELD_FLEXIBLE(id_)       , ::std::basic_string<unsigned char> id_ ## X_
#define FIELD_ARRAY(id_, ...)     , ::boost::container::vector< Cbpp ## id_ ## F_ > id_ ## X_
#define FIELD_LIST(id_, ...)      , ::boost::container::deque< Cbpp ## id_ ## F_ > id_ ## X_

MESSAGE_NAME::MESSAGE_NAME(STRIP_FIRST(void MESSAGE_FIELDS))
	: ::Poseidon::Cbpp::MessageBase()

#undef FIELD_VINT
#undef FIELD_VUINT
#undef FIELD_FIXED
#undef FIELD_STRING
#undef FIELD_BLOB
#undef FIELD_FLEXIBLE
#undef FIELD_ARRAY
#undef FIELD_LIST

#define FIELD_VINT(id_)           , id_(id_ ## X_)
#define FIELD_VUINT(id_)          , id_(id_ ## X_)
#define FIELD_FIXED(id_, n_)      , id_(id_ ## X_)
#define FIELD_STRING(id_)         , id_(STD_MOVE(id_ ## X_))
#define FIELD_BLOB(id_)           , id_(STD_MOVE(id_ ## X_))
#define FIELD_FLEXIBLE(id_)       , id_(STD_MOVE(id_ ## X_))
#define FIELD_ARRAY(id_, ...)     , id_(STD_MOVE(id_ ## X_))
#define FIELD_LIST(id_, ...)      , id_(STD_MOVE(id_ ## X_))

	MESSAGE_FIELDS
{ }

MESSAGE_NAME::MESSAGE_NAME(::Poseidon::StreamBuffer buffer_)
	: ::Poseidon::Cbpp::MessageBase()
{
	deserialize(buffer_);
	if(!buffer_.empty()){
		THROW_JUNK_AFTER_PACKET_(MESSAGE_NAME);
	}
}

MESSAGE_NAME::~MESSAGE_NAME(){ }

boost::uint64_t MESSAGE_NAME::get_id() const {
	return ID;
}
void MESSAGE_NAME::serialize(::Poseidon::StreamBuffer &buffer_) const {
	const AUTO(cur_, this);
	::Poseidon::StreamBuffer::WriteIterator w_(buffer_);

#undef FIELD_VINT
#undef FIELD_VUINT
#undef FIELD_FIXED
#undef FIELD_STRING
#undef FIELD_BLOB
#undef FIELD_FLEXIBLE
#undef FIELD_ARRAY
#undef FIELD_LIST

#define FIELD_VINT(id_)           {	\
                                    ::Poseidon::vint64_to_binary(cur_->id_, w_);	\
                                  }
#define FIELD_VUINT(id_)          {	\
                                    ::Poseidon::vuint64_to_binary(cur_->id_, w_);	\
                                  }
#define FIELD_FIXED(id_, n_)      {	\
                                    w_ = ::std::copy(cur_->id_.begin(), cur_->id_.end(), w_);	\
                                  }
#define FIELD_STRING(id_)         {	\
                                    ::Poseidon::vuint64_to_binary(cur_->id_.size(), w_);	\
                                    w_ = ::std::copy(cur_->id_.begin(), cur_->id_.end(), w_);	\
                                  }
#define FIELD_BLOB(id_)           {	\
                                    ::Poseidon::vuint64_to_binary(cur_->id_.size(), w_);	\
                                    w_ = ::std::copy(cur_->id_.begin(), cur_->id_.end(), w_);	\
                                  }
#define FIELD_FLEXIBLE(id_)       {	\
                                    w_ = ::std::copy(cur_->id_.begin(), cur_->id_.end(), w_);	\
                                  }
#define FIELD_ARRAY(id_, ...)     {	\
                                    ::Poseidon::vuint64_to_binary(cur_->id_.size(), w_);	\
                                    for(AUTO(it_, cur_->id_.begin()); it_ != cur_->id_.end(); ++it_){	\
                                      const AUTO(cur_, it_);	\
                                      __VA_ARGS__	\
                                    }	\
                                  }
#define FIELD_LIST(id_, ...)      {	\
                                    ::std::basic_string<unsigned char> chunk_;	\
                                    for(AUTO(it_, cur_->id_.begin()); it_ != cur_->id_.end(); ++it_){	\
                                      chunk_.clear();	\
                                      {	\
                                        const AUTO(cur_, it_);	\
                                        AUTO(w_, ::std::back_inserter(chunk_));	\
                                        __VA_ARGS__	\
                                      }	\
                                      ::Poseidon::vuint64_to_binary(chunk_.size(), w_);	\
                                      w_ = ::std::copy(chunk_.begin(), chunk_.end(), w_);	\
                                    }	\
                                    *w_ = 0;	\
                                    ++w_;	\
                                  }

	MESSAGE_FIELDS
}
void MESSAGE_NAME::deserialize(::Poseidon::StreamBuffer &buffer_){
	const AUTO(cur_, this);
	::Poseidon::StreamBuffer::ReadIterator r_(buffer_);

#undef FIELD_VINT
#undef FIELD_VUINT
#undef FIELD_FIXED
#undef FIELD_STRING
#undef FIELD_BLOB
#undef FIELD_FLEXIBLE
#undef FIELD_ARRAY
#undef FIELD_LIST

#define FIELD_VINT(id_)           {	\
                                    cur_->id_ = 0;	\
                                    if(!::Poseidon::vint64_from_binary(cur_->id_, r_, SIZE_MAX)){	\
                                      THROW_END_OF_STREAM_(MESSAGE_NAME, id_);	\
                                    }	\
                                  }
#define FIELD_VUINT(id_)          {	\
                                    cur_->id_ = 0;	\
                                    if(!::Poseidon::vuint64_from_binary(cur_->id_, r_, SIZE_MAX)){	\
                                      THROW_END_OF_STREAM_(MESSAGE_NAME, id_);	\
                                    }	\
                                  }
#define FIELD_FIXED(id_, n_)      {	\
                                    cur_->id_.fill(0);	\
                                    int c_;	\
                                    for(::std::size_t i_ = 0; i_ < n_; ++i_){	\
                                      if((c_ = *r_) < 0){	\
                                        THROW_END_OF_STREAM_(MESSAGE_NAME, id_);	\
                                      }	\
                                      ++r_;	\
                                      cur_->id_[i_] = static_cast<unsigned char>(c_);	\
                                    }	\
                                  }
#define FIELD_STRING(id_)         {	\
                                    cur_->id_.clear();	\
                                    ::boost::uint64_t n_;	\
                                    if(!::Poseidon::vuint64_from_binary(n_, r_, SIZE_MAX)){	\
                                      THROW_END_OF_STREAM_(MESSAGE_NAME, id_);	\
                                    }	\
                                    int c_;	\
                                    for(::std::size_t i_ = 0; i_ < n_; ++i_){	\
                                      if((c_ = *r_) < 0){	\
                                        THROW_END_OF_STREAM_(MESSAGE_NAME, id_);	\
                                      }	\
                                      ++r_;	\
                                      cur_->id_.push_back(c_);	\
                                    }	\
                                  }
#define FIELD_BLOB(id_)           {	\
                                    cur_->id_.clear();	\
                                    ::boost::uint64_t n_;	\
                                    if(!::Poseidon::vuint64_from_binary(n_, r_, SIZE_MAX)){	\
                                      THROW_END_OF_STREAM_(MESSAGE_NAME, id_);	\
                                    }	\
                                    int c_;	\
                                    for(::std::size_t i_ = 0; i_ < n_; ++i_){	\
                                      if((c_ = *r_) < 0){	\
                                        THROW_END_OF_STREAM_(MESSAGE_NAME, id_);	\
                                      }	\
                                      ++r_;	\
                                      cur_->id_.push_back(c_);	\
                                    }	\
                                  }
#define FIELD_FLEXIBLE(id_)       {	\
                                    cur_->id_.clear();	\
                                    int c_;	\
                                    while((c_ = *r_) >= 0){	\
                                      ++r_;	\
                                      cur_->id_.push_back(c_);	\
                                    }	\
                                  }
#define FIELD_ARRAY(id_, ...)     {	\
                                    cur_->id_.clear();	\
                                    ::boost::uint64_t n_;	\
                                    if(!::Poseidon::vuint64_from_binary(n_, r_, SIZE_MAX)){	\
                                      THROW_END_OF_STREAM_(MESSAGE_NAME, id_);	\
                                    }	\
                                    for(::std::size_t i_ = 0; i_ < n_; ++i_){	\
                                      const AUTO(it_, cur_->id_.emplace(cur_->id_.end()));	\
                                      const AUTO(cur_, it_);	\
                                      __VA_ARGS__	\
                                    }	\
                                  }
#define FIELD_LIST(id_, ...)      {	\
                                    cur_->id_.clear();	\
                                    ::boost::uint64_t n_;	\
                                    for(;;){	\
                                      if(!::Poseidon::vuint64_from_binary(n_, r_, SIZE_MAX)){	\
                                        THROW_END_OF_STREAM_(MESSAGE_NAME, id_);	\
                                      }	\
                                      if(n_ == 0){	\
                                        break;	\
                                      }	\
                                      ::Poseidon::StreamBuffer chunk_;	\
                                      int c_;	\
                                      for(::std::size_t i_ = 0; i_ < n_; ++i_){	\
                                        if((c_ = *r_) < 0){	\
                                          THROW_END_OF_STREAM_(MESSAGE_NAME, id_);	\
                                        }	\
                                        ++r_;	\
                                        chunk_.put(c_);	\
                                      }	\
                                      const AUTO(it_, cur_->id_.emplace(cur_->id_.end()));	\
                                      const AUTO(cur_, it_);	\
                                      ::Poseidon::StreamBuffer::ReadIterator r_(chunk_);	\
                                      __VA_ARGS__	\
                                    }	\
                                  }

	MESSAGE_FIELDS
}
void MESSAGE_NAME::dump_debug(::std::ostream &os_) const {
	static CONSTEXPR int s_indent_step_ = 2;

	const AUTO(cur_, this);
	int indent_ = s_indent_step_;

#undef FIELD_VINT
#undef FIELD_VUINT
#undef FIELD_FIXED
#undef FIELD_STRING
#undef FIELD_BLOB
#undef FIELD_FLEXIBLE
#undef FIELD_ARRAY
#undef FIELD_LIST

#define FIELD_VINT(id_)           {	\
                                    os_ << ::std::setw(indent_) <<"" <<TOKEN_TO_STR(id_) <<": vint = " <<cur_->id_ <<"\n";	\
                                  }
#define FIELD_VUINT(id_)          {	\
                                    os_ << ::std::setw(indent_) <<"" <<TOKEN_TO_STR(id_) <<": vuint = " <<cur_->id_ <<"\n";	\
                                  }
#define FIELD_FIXED(id_, n_)      {	\
                                    os_ << ::std::setw(indent_) <<"" <<TOKEN_TO_STR(id_) <<": fixed(" <<cur_->id_.size() <<") = " << ::Poseidon::HexPrinter(cur_->id_.data(), cur_->id_.size()) <<"\n";	\
                                  }
#define FIELD_STRING(id_)         {	\
                                    os_ << ::std::setw(indent_) <<"" <<TOKEN_TO_STR(id_) <<": string(" <<cur_->id_.size() <<") = " <<cur_->id_ <<"\n";	\
                                  }
#define FIELD_BLOB(id_)           {	\
                                    os_ << ::std::setw(indent_) <<"" <<TOKEN_TO_STR(id_) <<": blob(" <<cur_->id_.size() <<") = " << ::Poseidon::HexPrinter(cur_->id_.data(), cur_->id_.size()) <<"\n";	\
                                  }
#define FIELD_FLEXIBLE(id_)       {	\
                                    os_ << ::std::setw(indent_) <<"" <<TOKEN_TO_STR(id_) <<": flexible(" <<cur_->id_.size() <<") = " << ::Poseidon::HexPrinter(cur_->id_.data(), cur_->id_.size()) <<"\n";	\
                                  }
#define FIELD_ARRAY(id_, ...)     {	\
                                    os_ << ::std::setw(indent_) <<"" <<TOKEN_TO_STR(id_) <<": array(" <<cur_->id_.size() <<") = [\n";	\
                                    indent_ += s_indent_step_;	\
                                    for(AUTO(it_, cur_->id_.begin()); it_ != cur_->id_.end(); ++it_){	\
                                      const AUTO(cur_, it_);	\
                                      os_ << ::std::setw(indent_) <<"" <<"{\n";	\
                                      indent_ += s_indent_step_;	\
                                      __VA_ARGS__	\
                                      indent_ -= s_indent_step_;	\
                                      os_ << ::std::setw(indent_) <<"" <<"}\n";	\
                                    }	\
                                    indent_ -= s_indent_step_;	\
                                    os_ << ::std::setw(indent_) <<"" <<"]\n";	\
                                  }
#define FIELD_LIST(id_, ...)      {	\
                                    os_ << ::std::setw(indent_) <<"" <<TOKEN_TO_STR(id_) <<": list(" <<cur_->id_.size() <<") = [\n";	\
                                    indent_ += s_indent_step_;	\
                                    for(AUTO(it_, cur_->id_.begin()); it_ != cur_->id_.end(); ++it_){	\
                                      const AUTO(cur_, it_);	\
                                      os_ << ::std::setw(indent_) <<"" <<"{\n";	\
                                      indent_ += s_indent_step_;	\
                                      __VA_ARGS__	\
                                      indent_ -= s_indent_step_;	\
                                      os_ << ::std::setw(indent_) <<"" <<"}\n";	\
                                    }	\
                                    indent_ -= s_indent_step_;	\
                                    os_ << ::std::setw(indent_) <<"" <<"]\n";	\
                                  }

	os_ <<TOKEN_TO_STR(MESSAGE_NAME) <<"(" <<get_id() <<") = {\n";
	MESSAGE_FIELDS
	os_ <<"}\n";
}

#pragma GCC diagnostic pop
#endif // CBPP_MESSAGE_EMIT_EXTERNAL_DEFINITIONS

#undef MESSAGE_NAME
#undef MESSAGE_ID
#undef MESSAGE_FIELDS
