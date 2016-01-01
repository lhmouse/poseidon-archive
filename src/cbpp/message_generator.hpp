// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2016, LH_Mouse. All wrongs reserved.

#ifndef MESSAGE_NAME
#   error MESSAGE_NAME is undefined.
#endif

#ifndef MESSAGE_ID
#   error MESSAGE_ID is undefined.
#endif

#ifndef MESSAGE_FIELDS
#   error MESSAGE_FIELDS is undefined.
#endif

#ifndef POSEIDON_CBPP_MESSAGE_BASE_HPP_
#   error Please #include <poseidon/cbpp/message_base.hpp> first.
#endif

class MESSAGE_NAME : public ::Poseidon::Cbpp::MessageBase {
public:
	enum {
		ID = MESSAGE_ID
	};

public:

#undef FIELD_VINT
#undef FIELD_VUINT
#undef FIELD_STRING
#undef FIELD_BYTES
#undef FIELD_ARRAY

#define FIELD_VINT(name_)               ::boost::int64_t name_;
#define FIELD_VUINT(name_)              ::boost::uint64_t name_;
#define FIELD_STRING(name_)             ::std::string name_;
#define FIELD_BYTES(name_, size_)       ::boost::array<unsigned char, size_> name_;
#define FIELD_ARRAY(name_, fields_)     struct ElementOf ## name_ ## X_ {   \
											fields_ \
										};  \
										::std::vector<ElementOf ## name_ ## X_> name_;

	MESSAGE_FIELDS

public:

#undef FIELD_VINT
#undef FIELD_VUINT
#undef FIELD_STRING
#undef FIELD_BYTES
#undef FIELD_ARRAY

#define FIELD_VINT(name_)               + 1
#define FIELD_VUINT(name_)              + 1
#define FIELD_STRING(name_)             + 1
#define FIELD_BYTES(name_, size_)       + 1
#define FIELD_ARRAY(name_, fields_)

#if (0 MESSAGE_FIELDS) != 0
	MESSAGE_NAME()
		: ::Poseidon::Cbpp::MessageBase()

#undef FIELD_VINT
#undef FIELD_VUINT
#undef FIELD_STRING
#undef FIELD_BYTES
#undef FIELD_ARRAY

#define FIELD_VINT(name_)               , name_()
#define FIELD_VUINT(name_)              , name_()
#define FIELD_STRING(name_)             , name_()
#define FIELD_BYTES(name_, size_)       , name_()
#define FIELD_ARRAY(name_, fields_)     , name_()

		MESSAGE_FIELDS
	{
	}
#endif

#undef FIELD_VINT
#undef FIELD_VUINT
#undef FIELD_STRING
#undef FIELD_BYTES
#undef FIELD_ARRAY

#define FIELD_VINT(name_)               , ::boost::int64_t name_ ## X_
#define FIELD_VUINT(name_)              , ::boost::uint64_t name_ ## X_
#define FIELD_STRING(name_)             , ::std::string name_ ## X_
#define FIELD_BYTES(name_, size_),      , const ::boost::array<unsigned char, size_> &name_ ## X_
#define FIELD_ARRAY(name_, fields_)

	explicit MESSAGE_NAME(STRIP_FIRST(void MESSAGE_FIELDS))
		: ::Poseidon::Cbpp::MessageBase()

#undef FIELD_VINT
#undef FIELD_VUINT
#undef FIELD_STRING
#undef FIELD_BYTES
#undef FIELD_ARRAY

#define FIELD_VINT(name_)               , name_(name_ ## X_)
#define FIELD_VUINT(name_)              , name_(name_ ## X_)
#define FIELD_STRING(name_)             , name_(STD_MOVE(name_ ## X_))
#define FIELD_BYTES(name_, size_)       , name_(name_ ## X_)
#define FIELD_ARRAY(name_, fields_)     , name_()

		MESSAGE_FIELDS
	{
	}
	explicit MESSAGE_NAME(::Poseidon::StreamBuffer buffer_)
		: ::Poseidon::Cbpp::MessageBase()
	{
		*this << buffer_;

		if(!buffer_.empty()){
			THROW_JUNK_AFTER_PACKET_(MESSAGE_NAME);
		}
	}

public:
	void operator>>(::Poseidon::StreamBuffer &buffer_) const {
		::Poseidon::StreamBuffer::WriteIterator write_(buffer_);

		typedef MESSAGE_NAME Cur_;
		const Cur_ &cur_ = *this;

#undef FIELD_VINT
#undef FIELD_VUINT
#undef FIELD_STRING
#undef FIELD_BYTES
#undef FIELD_ARRAY

#define FIELD_VINT(name_)               ::Poseidon::vint50_to_binary(cur_.name_, write_);
#define FIELD_VUINT(name_)              ::Poseidon::vuint50_to_binary(cur_.name_, write_);
#define FIELD_STRING(name_)             ::Poseidon::vuint50_to_binary(cur_.name_.size(), write_);   \
										write_ = ::std::copy(cur_.name_.begin(), cur_.name_.end(), write_);
#define FIELD_BYTES(name_, size_)       write_ = ::std::copy(cur_.name_, cur_.name_ + size_, write_);
#define FIELD_ARRAY(name_, fields_)     ::Poseidon::vuint50_to_binary(cur_.name_.size(), write_);   \
										for(::boost::uint64_t i_ = 0; i_ < cur_.name_.size(); ++i_){    \
											typedef Cur_::ElementOf ## name_ ## X_ Element_;    \
											const Element_ &element_ = cur_.name_[i_];  \
											typedef Element_ Cur_;  \
											const Cur_ &cur_ = element_;    \
											\
											fields_ \
										}

		MESSAGE_FIELDS
	}

	void operator<<(::Poseidon::StreamBuffer &buffer_){
		::Poseidon::StreamBuffer::ReadIterator read_(buffer_);

		typedef MESSAGE_NAME Cur_;
		Cur_ &cur_ = *this;

#undef FIELD_VINT
#undef FIELD_VUINT
#undef FIELD_STRING
#undef FIELD_BYTES
#undef FIELD_ARRAY

#define FIELD_VINT(name_)               if(!::Poseidon::vint50_from_binary(cur_.name_, read_, buffer_.size())){ \
											THROW_END_OF_STREAM_(MESSAGE_NAME, name_);  \
										}
#define FIELD_VUINT(name_)              if(!::Poseidon::vuint50_from_binary(cur_.name_, read_, buffer_.size())){    \
											THROW_END_OF_STREAM_(MESSAGE_NAME, name_);  \
										}
#define FIELD_STRING(name_)             {   \
											::boost::uint64_t count_;   \
											if(!::Poseidon::vuint50_from_binary(count_, read_, buffer_.size())){    \
												THROW_END_OF_STREAM_(MESSAGE_NAME, name_);  \
											}   \
											if(buffer_.size() < count_){    \
												THROW_END_OF_STREAM_(MESSAGE_NAME, name_);  \
											}   \
											if(count_ > cur_.name_.max_size()){ \
												THROW_LENGTH_ERROR_(MESSAGE_NAME, name_);   \
											}   \
											/* cur_.name_.reserve(count_); */   \
											for(::boost::uint64_t i_ = 0; i_ < count_; ++i_){   \
												cur_.name_.push_back(*read_);   \
												++read_;    \
											}   \
										}
#define FIELD_BYTES(name_, size_)       if(buffer_.size() < size_){ \
											THROW_END_OF_STREAM_(MESSAGE_NAME, name_);  \
										}   \
										for(::boost::uint64_t i_ = 0; i_ < size_; ++i_){    \
											cur_.name_[i_] = *read_;    \
											++read_;    \
										}
#define FIELD_ARRAY(name_, fields_)     {   \
											::boost::uint64_t count_;   \
											if(!::Poseidon::vuint50_from_binary(count_, read_, buffer_.size())){    \
												THROW_END_OF_STREAM_(MESSAGE_NAME, name_);  \
											}   \
											/* cur_.name_.reserve(count_); */   \
											for(::boost::uint64_t i_ = 0; i_ < count_; ++i_){   \
												typedef Cur_::ElementOf ## name_ ## X_ Element_;    \
												cur_.name_.push_back(Element_());   \
												Element_ &element_ = cur_.name_.back(); \
												typedef Element_ Cur_;  \
												Cur_ &cur_ = element_;  \
												\
												fields_ \
											}   \
										}

		MESSAGE_FIELDS
	}

	operator ::Poseidon::StreamBuffer() const {
		::Poseidon::StreamBuffer buffer_;
		*this >> buffer_;
		return buffer_;
	}

	void dump_debug(::std::ostream &os_) const {
		typedef MESSAGE_NAME Cur_;
		const Cur_ &cur_ = *this;

		os_ <<TOKEN_TO_STR(MESSAGE_NAME) <<"(" <<ID <<") = {; ";

#undef FIELD_VINT
#undef FIELD_VUINT
#undef FIELD_STRING
#undef FIELD_BYTES
#undef FIELD_ARRAY

#define FIELD_VINT(name_)               os_ << #name_ <<" = " <<cur_.name_ <<"; ";
#define FIELD_VUINT(name_)              os_ << #name_ <<" = " <<cur_.name_ <<"; ";
#define FIELD_STRING(name_)             os_ << #name_ <<" = (" <<cur_.name_.size() <<")\"" <<cur_.name_ <<"\"; ";
#define FIELD_BYTES(name_, size_)       os_ << #name_ <<" = (" <<size_ <<")[ " << ::std::hex;   \
										for(::boost::uint64_t i_ = 0; i_ < size_; ++i_){    \
											os_ << ::std::setfill('0') << ::std::setw(2)    \
												<< static_cast<unsigned>(cur_.name_[i_]) <<' '; \
										}   \
										os_ << ::std::dec <<"]; ";
#define FIELD_ARRAY(name_, fields_)     os_ << #name_ <<" = (" <<cur_.name_.size() <<")[; ";    \
										for(::boost::uint64_t i_ = 0; i_ < cur_.name_.size(); ++i_){    \
											typedef Cur_::ElementOf ## name_ ## X_ Element_;    \
											const Element_ &element_ = cur_.name_[i_];  \
											typedef Element_ Cur_;  \
											const Cur_ &cur_ = element_;    \
											\
											os_ <<"{; ";    \
											fields_ \
											os_ <<"}; ";    \
										}   \
										os_ <<"]; ";

		MESSAGE_FIELDS

		os_ <<"}; ";
	}
};

inline ::Poseidon::StreamBuffer &operator<<(::Poseidon::StreamBuffer &buffer_, const MESSAGE_NAME &msg_){
	msg_ >>buffer_;
	return buffer_;
}
inline ::Poseidon::StreamBuffer &operator>>(::Poseidon::StreamBuffer &buffer_, MESSAGE_NAME &msg_){
	msg_ <<buffer_;
	return buffer_;
}

inline ::std::ostream &operator<<(::std::ostream &os_, const MESSAGE_NAME &msg_){
	msg_.dump_debug(os_);
	return os_;
}

#undef MESSAGE_NAME
#undef MESSAGE_ID
#undef MESSAGE_FIELDS
