#ifndef PROTOCOL_NAME
#	error PROTOCOL_NAME is undefined.
#endif

#ifndef PROTOCOL_FIELDS
#	error PROTOCOL_FIELDS is undefined.
#endif

#include "protocol_base.hpp"

#ifdef PROTOCOL_NAMESPACE
namespace PROTOCOL_NAMESPACE {
#endif

struct PROTOCOL_NAME : public ::Poseidon::ProtocolBase {

#undef FIELD_VINT50
#undef FIELD_VUINT50
#undef FIELD_STRING
#undef FIELD_ARRAY

#define FIELD_VINT50(name_)				long long name_;
#define FIELD_VUINT50(name_)			unsigned long long name_;
#define FIELD_STRING(name_)				::std::string name_;
#define FIELD_ARRAY(name_, fields_)		struct ElementOf ## name_ ## _ {	\
											fields_	\
										};	\
										::std::vector<ElementOf ## name_ ## _> name_;

	PROTOCOL_FIELDS

	PROTOCOL_NAME()
		: ProtocolBase()

#undef FIELD_VINT50
#undef FIELD_VUINT50
#undef FIELD_STRING
#undef FIELD_ARRAY

#define FIELD_VINT50(name_)				, name_()
#define FIELD_VUINT50(name_)			, name_()
#define FIELD_STRING(name_)				, name_()
#define FIELD_ARRAY(name_, fields_)		, name_()

		PROTOCOL_FIELDS
	{
	}

	void operator>>(::Poseidon::StreamBuffer &buffer_) const {
		typedef PROTOCOL_NAME Cur_;
		const Cur_ &cur_ = *this;
		::Poseidon::StreamBufferWriteIterator write_(buffer_);
		unsigned long long count_;

		(void)cur_;
		(void)write_;
		(void)count_;

#undef FIELD_VINT50
#undef FIELD_VUINT50
#undef FIELD_STRING
#undef FIELD_ARRAY

#define FIELD_VINT50(name_)				::Poseidon::vint50ToBinary(cur_.name_, write_);
#define FIELD_VUINT50(name_)			::Poseidon::vuint50ToBinary(cur_.name_, write_);
#define FIELD_STRING(name_)				::Poseidon::vuint50ToBinary(cur_.name_.size(), write_);	\
										write_ = ::std::copy(cur_.name_.begin(), cur_.name_.end(), write_);
#define FIELD_ARRAY(name_, fields_)		count_ = cur_.name_.size();	\
										::Poseidon::vuint50ToBinary(count_, write_);	\
										for(unsigned long long i = 0; i < count_; ++i){	\
											typedef Cur_::ElementOf ## name_ ## _ Element_;	\
											const Element_ &element_ = cur_.name_[i];	\
											typedef Element_ Cur_;	\
											const Cur_ &cur_ = element_;	\
											\
											fields_	\
										}

		PROTOCOL_FIELDS
	}

	void operator<<(::Poseidon::StreamBuffer &buffer_){
		typedef PROTOCOL_NAME Cur_;
		Cur_ &cur_ = *this;
		::Poseidon::StreamBufferReadIterator read_(buffer_);
		unsigned long long count_;

		(void)cur_;
		(void)read_;
		(void)count_;

#undef FIELD_VINT50
#undef FIELD_VUINT50
#undef FIELD_STRING
#undef FIELD_ARRAY

#define FIELD_VINT50(name_)				if(!::Poseidon::vint50FromBinary(cur_.name_, read_, buffer_.size())){	\
											THROW_EOS_;	\
										}
#define FIELD_VUINT50(name_)			if(!::Poseidon::vuint50FromBinary(cur_.name_, read_, buffer_.size())){	\
											THROW_EOS_;	\
										}
#define FIELD_STRING(name_)				unsigned long long count_;	\
										if(!::Poseidon::vuint50FromBinary(count_, read_, buffer_.size())){	\
											THROW_EOS_;	\
										}	\
										if(buffer_.size() < count_){	\
											THROW_EOS_;	\
										}	\
										cur_.name_.resize(count_);	\
										for(unsigned long long i = 0; i < count_; ++i){	\
											cur_.name_[i] = buffer_.get();	\
										}
#define FIELD_ARRAY(name_, fields_)		if(!::Poseidon::vuint50FromBinary(count_, read_, buffer_.size())){	\
											THROW_EOS_;  \
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
										}

		PROTOCOL_FIELDS
	}
};

#ifdef PROTOCOL_NAMESPACE
}
#endif

#undef PROTOCOL_NAME
#undef PROTOCOL_FIELDS
