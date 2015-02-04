// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2015, LH_Mouse. All wrongs reserved.

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

class PROTOCOL_NAME : public ::Poseidon::PlayerProtocolBase {
public:
	enum {
		ID = PROTOCOL_ID
	};

private:

#undef FIELD_VINT
#undef FIELD_VUINT
#undef FIELD_STRING
#undef FIELD_BYTES
#undef FIELD_ARRAY

#define FIELD_VINT(name_)				long long name_;
#define FIELD_VUINT(name_)				unsigned long long name_;
#define FIELD_STRING(name_)				::std::string name_;
#define FIELD_BYTES(name_, size_)		unsigned char name_[size_];
#define FIELD_ARRAY(name_, fields_)		struct ElementOf ## name_ ## X_ {	\
											fields_	\
										};	\
										::std::vector<ElementOf ## name_ ## X_> name_;

	PROTOCOL_FIELDS

public:

#undef FIELD_VINT
#undef FIELD_VUINT
#undef FIELD_STRING
#undef FIELD_BYTES
#undef FIELD_ARRAY

#define FIELD_VINT(name_)				+ 1
#define FIELD_VUINT(name_)				+ 1
#define FIELD_STRING(name_)				+ 1
#define FIELD_BYTES(name_, size_)
#define FIELD_ARRAY(name_, fields_)

#if (0 PROTOCOL_FIELDS) != 0
	PROTOCOL_NAME()
		: ::Poseidon::PlayerProtocolBase()

#undef FIELD_VINT
#undef FIELD_VUINT
#undef FIELD_STRING
#undef FIELD_BYTES
#undef FIELD_ARRAY

#define FIELD_VINT(name_)				, name_()
#define FIELD_VUINT(name_)				, name_()
#define FIELD_STRING(name_)				, name_()
#define FIELD_BYTES(name_, size_)		, name_()
#define FIELD_ARRAY(name_, fields_)		, name_()

		PROTOCOL_FIELDS
	{
	}
#endif

#undef FIELD_VINT
#undef FIELD_VUINT
#undef FIELD_STRING
#undef FIELD_BYTES
#undef FIELD_ARRAY

#define FIELD_VINT(name_)				, long long name_ ## X_
#define FIELD_VUINT(name_)				, unsigned long long name_ ## X_
#define FIELD_STRING(name_)				, ::std::string name_ ## X_
#define FIELD_BYTES(name_, size_)
#define FIELD_ARRAY(name_, fields_)

	explicit PROTOCOL_NAME(STRIP_FIRST(void PROTOCOL_FIELDS))
		: ::Poseidon::PlayerProtocolBase()

#undef FIELD_VINT
#undef FIELD_VUINT
#undef FIELD_STRING
#undef FIELD_BYTES
#undef FIELD_ARRAY

#define FIELD_VINT(name_)				, name_(name_ ## X_)
#define FIELD_VUINT(name_)				, name_(name_ ## X_)
#define FIELD_STRING(name_)				, name_(STD_MOVE(name_ ## X_))
#define FIELD_BYTES(name_, size_)		, name_()
#define FIELD_ARRAY(name_, fields_)		, name_()

		PROTOCOL_FIELDS
	{
	}
	explicit PROTOCOL_NAME(::Poseidon::StreamBuffer &buffer_)
		: ::Poseidon::PlayerProtocolBase()
	{
		*this << buffer_;
	}

public:

#undef FIELD_VINT
#undef FIELD_VUINT
#undef FIELD_STRING
#undef FIELD_BYTES
#undef FIELD_ARRAY

#define FIELD_VINT(name_)	\
	const long long & get_ ## name_() const {	\
		return name_;	\
	}	\
	long long & get_ ## name_(){	\
		return name_;	\
	}	\
	void set_ ## name_(long long val_){	\
		name_ = val_;	\
	}

#define FIELD_VUINT(name_)	\
	const unsigned long long & get_ ## name_() const {	\
		return name_;	\
	}	\
	unsigned long long & get_ ## name_(){	\
		return name_;	\
	}	\
	void set_ ## name_(unsigned long long val_){	\
		name_ = val_;	\
	}

#define FIELD_STRING(name_)	\
	const std::string & get_ ## name_() const {	\
		return name_;	\
	}	\
	std::string & get_ ## name_(){	\
		return name_;	\
	}	\
	void set_ ## name_(std::string val_){	\
		name_.swap(val_);	\
	}

#define FIELD_BYTES(name_, size_)	\
	const unsigned char (& get_ ## name_())[size_] const {	\
		return name_;	\
	}	\
	unsigned char (& get_ ## name_())[size_] {	\
		return name_;	\
	}

#define FIELD_ARRAY(name_, fields_)	\
	const ::std::vector<ElementOf ## name_ ## X_> & get_ ## name_() const {	\
		return name_;	\
	}	\
	::std::vector<ElementOf ## name_ ## X_> & get_ ## name_(){	\
		return name_;	\
	}

	PROTOCOL_FIELDS

	void operator>>(::Poseidon::StreamBuffer &buffer_) const {
		::Poseidon::StreamBuffer::WriteIterator write_(buffer_);

		typedef PROTOCOL_NAME Cur_;
		const Cur_ &cur_ = *this;

#undef FIELD_VINT
#undef FIELD_VUINT
#undef FIELD_STRING
#undef FIELD_BYTES
#undef FIELD_ARRAY

#define FIELD_VINT(name_)				::Poseidon::vint50ToBinary(cur_.name_, write_);
#define FIELD_VUINT(name_)				::Poseidon::vuint50ToBinary(cur_.name_, write_);
#define FIELD_STRING(name_)				{	\
											::Poseidon::vuint50ToBinary(cur_.name_.size(), write_);	\
											write_ = ::std::copy(	\
												cur_.name_.begin(), cur_.name_.end(), write_);	\
										}
#define FIELD_BYTES(name_, size_)		write_ = ::std::copy(cur_.name_, cur_name_ + size_, write_);
#define FIELD_ARRAY(name_, fields_)		{	\
											const unsigned long long count_ = cur_.name_.size();	\
											::Poseidon::vuint50ToBinary(count_, write_);	\
											for(unsigned long long i_ = 0; i_ < count_; ++i_){	\
												typedef Cur_::ElementOf ## name_ ## X_ Element_;	\
												const Element_ &element_ = cur_.name_[i_];	\
												typedef Element_ Cur_;	\
												const Cur_ &cur_ = element_;	\
												\
												fields_	\
											}	\
										}

		PROTOCOL_FIELDS
	}

	void operator<<(::Poseidon::StreamBuffer &buffer_){
		::Poseidon::StreamBuffer::ReadIterator read_(buffer_);

		typedef PROTOCOL_NAME Cur_;
		Cur_ &cur_ = *this;

#undef FIELD_VINT
#undef FIELD_VUINT
#undef FIELD_STRING
#undef FIELD_BYTES
#undef FIELD_ARRAY

#define FIELD_VINT(name_)				if(!::Poseidon::vint50FromBinary(cur_.name_, read_, buffer_.size())){	\
											THROW_END_OF_STREAM_(PROTOCOL_NAME, name_);	\
										}
#define FIELD_VUINT(name_)				if(!::Poseidon::vuint50FromBinary(cur_.name_, read_, buffer_.size())){	\
											THROW_END_OF_STREAM_(PROTOCOL_NAME, name_);	\
										}
#define FIELD_STRING(name_)				{	\
											unsigned long long count_;	\
											if(!::Poseidon::vuint50FromBinary(count_, read_, buffer_.size())){	\
												THROW_END_OF_STREAM_(PROTOCOL_NAME, name_);	\
											}	\
											if(buffer_.size() < count_){	\
												THROW_END_OF_STREAM_(PROTOCOL_NAME, name_);	\
											}	\
											for(unsigned long long i_ = 0; i_ < count_; ++i_){	\
												cur_.name_.push_back(*read_);	\
												++read_;	\
											}	\
										}
#define FIELD_BYTES(name_, size_)		if(buffer_.size() < size_){	\
											THROW_END_OF_STREAM_(PROTOCOL_NAME, name_);	\
										}	\
										for(::std::size_t i_ = 0; i_ < size_; ++i_){	\
											name_[i_] = *read_;	\
											++read_;	\
										}
#define FIELD_ARRAY(name_, fields_)		{	\
											unsigned long long count_;	\
											if(!::Poseidon::vuint50FromBinary(count_, read_, buffer_.size())){	\
												THROW_END_OF_STREAM_(PROTOCOL_NAME, name_);	\
											}	\
											cur_.name_.clear();	\
											for(unsigned long long i_ = 0; i_ < count_; ++i_){	\
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

		if(!buffer_.empty()){
			THROW_JUNK_AFTER_PACKET_(PROTOCOL_NAME);
		}
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
