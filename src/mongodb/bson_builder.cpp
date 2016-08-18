// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2016, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "bson_builder.hpp"
#include "oid.hpp"
#include "../uuid.hpp"
#include "../raii.hpp"
#include "../protocol_exception.hpp"
#include <mongo-client/bson.h>
#include <boost/static_assert.hpp>
#include <boost/type_traits/is_pod.hpp>

namespace Poseidon {

namespace {
	struct BsonDeleter {
		CONSTEXPR ::bson *operator()() const NOEXCEPT {
			return NULLPTR;
		}
		void operator()(::bson *b) const NOEXCEPT {
			::bson_free(b);
		}
	};

	template<typename T>
	T aliased_load(const void *ptr){
		BOOST_STATIC_ASSERT(boost::is_pod<T>::value);
		T value;
		std::memcpy(&value, ptr, sizeof(value));
		return value;
	}
	template<typename T>
	void aliased_store(void *ptr, const T &value){
		BOOST_STATIC_ASSERT(boost::is_pod<T>::value);
		std::memcpy(ptr, &value, sizeof(value));
	}
	template<typename T>
	::gint32 narrowing_cast_int32(const T &value){
		if(value > 0x7FFFFFFF){
			DEBUG_THROW(ProtocolException,
				sslit("The value does not fit into a signed 32-bit integer"), -1);
		}
		return static_cast< ::gint32>(value);
	}
}

namespace MongoDb {
	void BsonBuilder::append_oid(const Oid &oid){
		Element elem = { T_OID, sslit("_id") };
		std::memcpy(elem.small, oid.data(), oid.size());
		m_queue.push_back(STD_MOVE(elem));
	}
	void BsonBuilder::append_boolean(SharedNts name, bool value){
		Element elem = { T_BOOLEAN, STD_MOVE(name) };
		aliased_store(elem.small, value);
		m_queue.push_back(STD_MOVE(elem));
	}
	void BsonBuilder::append_signed(SharedNts name, boost::int64_t value){
		Element elem = { T_SIGNED, STD_MOVE(name) };
		aliased_store(elem.small, value);
		m_queue.push_back(STD_MOVE(elem));
	}
	void BsonBuilder::append_unsigned(SharedNts name, boost::uint64_t value){
		Element elem = { T_UNSIGNED, STD_MOVE(name) };
		aliased_store(elem.small, value);
		m_queue.push_back(STD_MOVE(elem));
	}
	void BsonBuilder::append_double(SharedNts name, double value){
		Element elem = { T_DOUBLE, STD_MOVE(name) };
		aliased_store(elem.small, value);
		m_queue.push_back(STD_MOVE(elem));
	}
	void BsonBuilder::append_string(SharedNts name, std::string value){
		Element elem = { T_STRING, STD_MOVE(name) };
		elem.large.swap(value);
		m_queue.push_back(STD_MOVE(elem));
	}
	void BsonBuilder::append_datetime(SharedNts name, boost::uint64_t value){
		Element elem = { T_DATETIME, STD_MOVE(name) };
		aliased_store(elem.small, value);
		m_queue.push_back(STD_MOVE(elem));
	}
	void BsonBuilder::append_uuid(SharedNts name, const Uuid &value){
		Element elem = { T_UUID, STD_MOVE(name) };
		value.to_string(elem.small);
		m_queue.push_back(STD_MOVE(elem));
	}
	void BsonBuilder::append_blob(SharedNts name, std::string value){
		Element elem = { T_BLOB, STD_MOVE(name) };
		elem.large.swap(value);
		m_queue.push_back(STD_MOVE(elem));
	}

	void BsonBuilder::append_regex(SharedNts name, std::string value, const char *options){
		Element elem = { T_REGEX, STD_MOVE(name) };
		elem.large.swap(value);
		std::size_t options_len = 0;
		if(options){
			options_len = std::strlen(options);
		}
		if(options_len > 0){
			if(options_len > sizeof(elem.small) - 1){
				options_len = sizeof(elem.small) - 1;
			}
			std::memcpy(elem.small, options, options_len);
		}
		elem.small[options_len] = 0;
		m_queue.push_back(STD_MOVE(elem));
	}
	void BsonBuilder::append_object(SharedNts name, const BsonBuilder &value){
		Element elem = { T_OBJECT, STD_MOVE(name) };
		std::string str = value.build();
		if(!str.empty()){
			str.erase(str.end() - 1);
		}
		elem.large.swap(str);
		m_queue.push_back(STD_MOVE(elem));
	}
	void BsonBuilder::append_array(SharedNts name, const BsonBuilder &value){
		Element elem = { T_ARRAY, STD_MOVE(name) };
		std::string str = value.build();
		if(!str.empty()){
			str.erase(str.end() - 1);
		}
		elem.large.swap(str);
		m_queue.push_back(STD_MOVE(elem));
	}

	void BsonBuilder::build(std::string &str) const {
		const UniqueHandle<BsonDeleter> b(::bson_new());
		if(!b){
			DEBUG_THROW(ProtocolException,
				sslit("Failed to create BSON object: bson_new() failed"), -1);
		}
		UniqueHandle<BsonDeleter> temp;
		for(AUTO(it, m_queue.begin()); it != m_queue.end(); ++it){
			switch(it->type){
				const ::guint8 *ptr_byte;
				bool val_bool;
				boost::int64_t val_int64;
				boost::uint64_t val_uint64;
				double val_double;
				::gint32 val_int32;

			case T_OID:
				ptr_byte = reinterpret_cast<const ::guint8 *>(it->small);
				if(!::bson_append_oid(b.get(), it->name.get(), ptr_byte)){
					DEBUG_THROW(ProtocolException,
						sslit("Failed to append data to BSON object: bson_append_oid() failed"), -1);
				}
				break;

			case T_BOOLEAN:
				val_bool = aliased_load<bool>(it->small);
				if(val_bool == false){
					break;
				}
				if(!::bson_append_boolean(b.get(), it->name.get(), val_bool)){
					DEBUG_THROW(ProtocolException,
						sslit("Failed to append data to BSON object: bson_append_boolean() failed"), -1);
				}
				break;

			case T_SIGNED:
				val_int64 = aliased_load<boost::int64_t>(it->small);
				if(val_int64 == 0){
					break;
				}
				if(!::bson_append_int64(b.get(), it->name.get(), val_int64)){
					DEBUG_THROW(ProtocolException,
						sslit("Failed to append data to BSON object: bson_append_int64() failed"), -1);
				}
				break;

			case T_UNSIGNED:
				val_uint64 = aliased_load<boost::uint64_t>(it->small);
				if(val_uint64 == 0){
					break;
				}
				if(!::bson_append_int64(b.get(), it->name.get(), static_cast< ::gint64>(val_uint64))){
					DEBUG_THROW(ProtocolException,
						sslit("Failed to append data to BSON object: bson_append_int64() failed"), -1);
				}
				break;

			case T_DOUBLE:
				val_double = aliased_load<double>(it->small);
				if(val_double == 0){
					break;
				}
				if(!::bson_append_double(b.get(), it->name.get(), val_double)){
					DEBUG_THROW(ProtocolException,
						sslit("Failed to append data to BSON object: bson_append_double() failed"), -1);
				}
				break;

			case T_STRING:
				val_int32 = narrowing_cast_int32(it->large.size());
				if(val_int32 == 0){
					break;
				}
				if(!::bson_append_string(b.get(), it->name.get(), it->large.data(), val_int32)){
					DEBUG_THROW(ProtocolException,
						sslit("Failed to append data to BSON object: bson_append_string() failed"), -1);
				}
				break;

			case T_DATETIME:
				val_uint64 = aliased_load<boost::uint64_t>(it->small);
				if(val_uint64 == 0){
					break;
				}
				if(!::bson_append_utc_datetime(b.get(), it->name.get(), static_cast< ::gint64>(val_uint64))){
					DEBUG_THROW(ProtocolException,
						sslit("Failed to append data to BSON object: bson_append_utc_datetime() failed"), -1);
				}
				break;

			case T_UUID:
				if(!::bson_append_string(b.get(), it->name.get(), it->small, 36)){
					DEBUG_THROW(ProtocolException,
						sslit("Failed to append data to BSON object: bson_append_string() failed"), -1);
				}
				break;

			case T_BLOB:
				ptr_byte = reinterpret_cast<const ::guint8 *>(it->large.data());
				val_int32 = narrowing_cast_int32(it->large.size());
				if(val_int32 == 0){
					break;
				}
				if(!::bson_append_binary(b.get(), it->name.get(), BSON_BINARY_SUBTYPE_USER_DEFINED, ptr_byte, val_int32)){
					DEBUG_THROW(ProtocolException,
						sslit("Failed to append data to BSON object: bson_append_binary() failed"), -1);
				}
				break;

			case T_REGEX:
				val_int32 = narrowing_cast_int32(it->large.size());
				if(val_int32 == 0){
					break;
				}
				if(!::bson_append_regex(b.get(), it->name.get(), it->large.c_str(), it->small)){
					DEBUG_THROW(ProtocolException,
						sslit("Failed to append data to BSON object: bson_append_regex() failed"), -1);
				}
				break;

			case T_OBJECT:
				ptr_byte = reinterpret_cast<const ::guint8 *>(it->large.data());
				val_int32 = narrowing_cast_int32(it->large.size());
				if(val_int32 == 0){
					break;
				}
				if(!temp.reset(::bson_new_from_data(ptr_byte, val_int32))){
					DEBUG_THROW(ProtocolException,
						sslit("Failed to create BSON object: bson_new() failed"), -1);
				}
				if(!::bson_finish(temp.get())){
					DEBUG_THROW(ProtocolException,
						sslit("Failed to finish BSON object: bson_finish() failed"), -1);
				}
				if(!::bson_append_document(b.get(), it->name.get(), temp.get())){
					DEBUG_THROW(ProtocolException,
						sslit("Failed to append data to BSON object: bson_append_document() failed"), -1);
				}
				break;

			case T_ARRAY:
				ptr_byte = reinterpret_cast<const ::guint8 *>(it->large.data());
				val_int32 = narrowing_cast_int32(it->large.size());
				if(val_int32 == 0){
					break;
				}
				if(!temp.reset(::bson_new_from_data(ptr_byte, val_int32))){
					DEBUG_THROW(ProtocolException,
						sslit("Failed to create BSON object: bson_new() failed"), -1);
				}
				if(!::bson_finish(temp.get())){
					DEBUG_THROW(ProtocolException,
						sslit("Failed to finish BSON object: bson_finish() failed"), -1);
				}
				if(!::bson_append_array(b.get(), it->name.get(), temp.get())){
					DEBUG_THROW(ProtocolException,
						sslit("Failed to append data to BSON object: bson_append_document() failed"), -1);
				}
				break;

			default:
				DEBUG_THROW(ProtocolException,
					sslit("Failed to append data to BSON object: Unknown BSON element type"), -1);
			}
		}
		if(!::bson_finish(b.get())){
			DEBUG_THROW(ProtocolException,
				sslit("Failed to finish BSON object: bson_finish() failed"), -1);
		}
		str.assign(reinterpret_cast<const char *>(::bson_data(b.get())), static_cast<std::size_t>(::bson_size(b.get())));
	}
}

}
