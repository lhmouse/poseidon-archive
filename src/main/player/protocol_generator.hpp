#ifndef PROTOCOL_NAME
#	error PROTOCOL_NAME is undefined.
#endif

#ifndef PROTOCOL_ID
#	error PROTOCOL_ID is undefined.
#endif

#ifndef PROTOCOL_FIELDS
#	error PROTOCOL_FIELDS is undefined.
#endif

#ifndef POSEIDON_PLAYER_PROTOCOL_BASE_HPP_
#   error Please #include "protocol_base.hpp" first.
#endif

#include "../cxx_util.hpp"

struct PROTOCOL_NAME : public ::Poseidon::ProtocolBase {
	enum {
		ID = PROTOCOL_ID
	};

#undef FIELD_VINT
#undef FIELD_VUINT
#undef FIELD_BYTES
#undef FIELD_STRING
#undef FIELD_ARRAY

#define FIELD_VINT(name_)				long long name_;
#define FIELD_VUINT(name_)				unsigned long long name_;
#define FIELD_BYTES(name_, size_)		unsigned char name_[size_];
#define FIELD_STRING(name_)				::std::string name_;
#define FIELD_ARRAY(name_, fields_)		struct ElementOf ## name_ ## X_ {	\
											fields_	\
										};	\
										::std::vector<ElementOf ## name_ ## X_> name_;

	PROTOCOL_FIELDS

#undef FIELD_VINT
#undef FIELD_VUINT
#undef FIELD_BYTES
#undef FIELD_STRING
#undef FIELD_ARRAY

#define FIELD_VINT(name_)				, long long name_ ## X_ = 0
#define FIELD_VUINT(name_)				, unsigned long long name_ ## X_ = 0
#define FIELD_BYTES(name_, size_)
#define FIELD_STRING(name_)				, ::std::string name_ ## X_ = ::std::string()
#define FIELD_ARRAY(name_, fields_)

	explicit PROTOCOL_NAME(STRIP_FIRST(void PROTOCOL_FIELDS))
		: ProtocolBase()

#undef FIELD_VINT
#undef FIELD_VUINT
#undef FIELD_BYTES
#undef FIELD_STRING
#undef FIELD_ARRAY

#define FIELD_VINT(name_)				, name_(name_ ## X_)
#define FIELD_VUINT(name_)				, name_(name_ ## X_)
#define FIELD_BYTES(name_, size_)		, name_()
#define FIELD_STRING(name_)				, name_(STD_MOVE(name_ ## X_))
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
												typedef Cur_::ElementOf ## name_ ## X_ Element_;	\
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
												typedef Cur_::ElementOf ## name_ ## X_ Element_;	\
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

#undef PROTOCOL_NAME
#undef PROTOCOL_ID
#undef PROTOCOL_FIELDS
