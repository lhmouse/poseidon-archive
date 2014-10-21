#ifndef PROTOCOL_NAME
#	error PROTOCOL_NAME is undefined.
#endif

#ifndef PROTOCOL_FIELDS
#	error PROTOCOL_FIELDS is undefined.
#endif

#ifndef POSEIDON_PLAYER_PROTOCOL_BASE_HPP_
#   error Please #include "protocol_base.hpp" first.
#endif

#ifndef POSEIDON_PLAYER_PROTOCOL_GENERATOR_HPP_
#define POSEIDON_PLAYER_PROTOCOL_GENERATOR_HPP_

#define PROTOCOL_STRIP_FIRST_2_(_, ...)		__VA_ARGS__
#define PROTOCOL_STRIP_FIRST_(...)			PROTOCOL_STRIP_FIRST_2_(__VA_ARGS__)

#endif

#ifdef PROTOCOL_NAMESPACE
namespace PROTOCOL_NAMESPACE {
#endif

struct PROTOCOL_NAME : public ::Poseidon::ProtocolBase {

#undef FIELD_VINT
#undef FIELD_VUINT
#undef FIELD_BYTES
#undef FIELD_STRING
#undef FIELD_ARRAY

#define FIELD_VINT(name_)				long long name_;
#define FIELD_VUINT(name_)				unsigned long long name_;
#define FIELD_BYTES(name_, size_)		unsigned char name_[size_];
#define FIELD_STRING(name_)				::std::string name_;
#define FIELD_ARRAY(name_, fields_)		struct ElementOf ## name_ ## _ {	\
											fields_	\
										};	\
										::std::vector<ElementOf ## name_ ## _> name_;

	PROTOCOL_FIELDS

#undef FIELD_VINT
#undef FIELD_VUINT
#undef FIELD_BYTES
#undef FIELD_STRING
#undef FIELD_ARRAY

#define FIELD_VINT(name_)				, long long name_ ## _ = VAL_INIT
#define FIELD_VUINT(name_)				, unsigned long long name_ ## _ = VAL_INIT
#define FIELD_BYTES(name_, size_)
#define FIELD_STRING(name_)				, ::std::string name_ ## _ = VAL_INIT
#define FIELD_ARRAY(name_, fields_)

	explicit PROTOCOL_NAME(PROTOCOL_STRIP_FIRST_(void PROTOCOL_FIELDS))
		: ProtocolBase()

#undef FIELD_VINT
#undef FIELD_VUINT
#undef FIELD_BYTES
#undef FIELD_STRING
#undef FIELD_ARRAY

#define FIELD_VINT(name_)				, name_(STD_MOVE(name_ ## _))
#define FIELD_VUINT(name_)				, name_(STD_MOVE(name_ ## _))
#define FIELD_BYTES(name_, size_)		, name_()
#define FIELD_STRING(name_)				, name_(STD_MOVE(name_ ## _))
#define FIELD_ARRAY(name_, fields_)		, name_()

		PROTOCOL_FIELDS
	{
	}
	explicit PROTOCOL_NAME(::Poseidon::StreamBuffer &buffer_)
		: ProtocolBase()
	{
		*this << buffer_;
	}

	void operator>>(::Poseidon::StreamBuffer &buffer_) const {
		::Poseidon::StreamBufferWriteIterator write_(buffer_);

		typedef PROTOCOL_NAME Cur_;
		const Cur_ &cur_ = *this;

#undef FIELD_VINT
#undef FIELD_VUINT
#undef FIELD_BYTES
#undef FIELD_STRING
#undef FIELD_ARRAY

#define FIELD_VINT(name_)				::Poseidon::vint50ToBinary(cur_.name_, write_);
#define FIELD_VUINT(name_)				::Poseidon::vuint50ToBinary(cur_.name_, write_);
#define FIELD_BYTES(name_, size_)		write_ = ::std::copy(cur_.name_, cur_name_ + size_, write_);
#define FIELD_STRING(name_)				::Poseidon::vuint50ToBinary(cur_.name_.size(), write_);	\
										write_ = ::std::copy(cur_.name_.begin(), cur_.name_.end(), write_);
#define FIELD_ARRAY(name_, fields_)		{	\
											const unsigned long long count_ = cur_.name_.size();	\
											::Poseidon::vuint50ToBinary(count_, write_);	\
											for(unsigned long long i = 0; i < count_; ++i){	\
												typedef Cur_::ElementOf ## name_ ## _ Element_;	\
												const Element_ &element_ = cur_.name_[i];	\
												typedef Element_ Cur_;	\
												const Cur_ &cur_ = element_;	\
												\
												fields_	\
											}	\
										}

		PROTOCOL_FIELDS
	}

	void operator<<(::Poseidon::StreamBuffer &buffer_){
		::Poseidon::StreamBufferReadIterator read_(buffer_);

		typedef PROTOCOL_NAME Cur_;
		Cur_ &cur_ = *this;

#undef FIELD_VINT
#undef FIELD_VUINT
#undef FIELD_BYTES
#undef FIELD_STRING
#undef FIELD_ARRAY

#define FIELD_VINT(name_)				if(!::Poseidon::vint50FromBinary(cur_.name_, read_, buffer_.size())){	\
											THROW_EOS_;	\
										}
#define FIELD_VUINT(name_)				if(!::Poseidon::vuint50FromBinary(cur_.name_, read_, buffer_.size())){	\
											THROW_EOS_;	\
										}
#define FIELD_BYTES(name_, size_)		if(buffer_.size() < size_){	\
											THROW_EOS_;	\
										}	\
										buffer_.get(cur_.name_, size_);
#define FIELD_STRING(name_)				{	\
											unsigned long long count_;	\
											if(!::Poseidon::vuint50FromBinary(count_, read_, buffer_.size())){	\
												THROW_EOS_;	\
											}	\
											if(buffer_.size() < count_){	\
												THROW_EOS_;	\
											}	\
											for(unsigned long long i = 0; i < count_; ++i){	\
												cur_.name_.push_back(buffer_.get());	\
											}	\
										}
#define FIELD_ARRAY(name_, fields_)		{	\
											unsigned long long count_;	\
											if(!::Poseidon::vuint50FromBinary(count_, read_, buffer_.size())){	\
												THROW_EOS_;	\
											}	\
											cur_.name_.clear();	\
											for(unsigned long long i = 0; i < count_; ++i){	\
												typedef Cur_::ElementOf ## name_ ## _ Element_;	\
												cur_.name_.push_back(Element_());	\
												Element_ &element_ = cur_.name_.back();	\
												typedef Element_ Cur_;	\
												Cur_ &cur_ = element_;	\
												\
												fields_	\
											}	\
										}

		PROTOCOL_FIELDS
	}

	operator ::Poseidon::StreamBuffer() const {
		::Poseidon::StreamBuffer buffer_;
		*this >> buffer_;
		return buffer_;
	}
};

#ifdef PROTOCOL_NAMESPACE
}
#endif

#undef PROTOCOL_NAME
#undef PROTOCOL_FIELDS
