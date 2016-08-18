// 这个文件是 Poseidon 服务器应用程序框架的一部分。
// Copyleft 2014 - 2016, LH_Mouse. All wrongs reserved.

#include "../precompiled.hpp"
#include "bson_builder.hpp"
#include "oid.hpp"
#include "../uuid.hpp"
#include "../raii.hpp"
#include "../protocol_exception.hpp"
#include <mongo-client/bson.h>

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
}

namespace MongoDb {
	void BsonBuilder::append_oid(const Oid &oid){
		Element elem = { T_OID, sslit("_id") };
		std::memcpy(elem.small, oid.data(), oid.size());
		m_queue.push_back(STD_MOVE(elem));
	}
	void BsonBuilder::append_boolean(SharedNts name, bool value){
		Element elem = { T_BOOLEAN, STD_MOVE(name) };
		std::memcpy(elem.small, &value, sizeof(value));
		m_queue.push_back(STD_MOVE(elem));
	}
	void BsonBuilder::append_signed(SharedNts name, boost::int64_t value){
		Element elem = { T_SIGNED, STD_MOVE(name) };
		std::memcpy(elem.small, &value, sizeof(value));
		m_queue.push_back(STD_MOVE(elem));
	}
	void BsonBuilder::append_unsigned(SharedNts name, boost::uint64_t value){
		Element elem = { T_UNSIGNED, STD_MOVE(name) };
		std::memcpy(elem.small, &value, sizeof(value));
		m_queue.push_back(STD_MOVE(elem));
	}
	void BsonBuilder::append_double(SharedNts name, double value){
		Element elem = { T_DOUBLE, STD_MOVE(name) };
		std::memcpy(elem.small, &value, sizeof(value));
		m_queue.push_back(STD_MOVE(elem));
	}
	void BsonBuilder::append_string(SharedNts name, std::string value){
		Element elem = { T_STRING, STD_MOVE(name) };
		elem.large.swap(value);
		m_queue.push_back(STD_MOVE(elem));
	}
	void BsonBuilder::append_datetime(SharedNts name, boost::uint64_t value){
		Element elem = { T_STRING, STD_MOVE(name) };
		std::memcpy(elem.small, &value, sizeof(value));
		m_queue.push_back(STD_MOVE(elem));
	}
	void BsonBuilder::append_uuid(SharedNts name, const Uuid &value){
		Element elem = { T_STRING, STD_MOVE(name) };
		value.to_string(elem.small);
		m_queue.push_back(STD_MOVE(elem));
	}
	void BsonBuilder::append_blob(SharedNts name, std::string value){
		Element elem = { T_BLOB, STD_MOVE(name) };
		elem.large.swap(value);
		m_queue.push_back(STD_MOVE(elem));
	}
	void BsonBuilder::append_regex(SharedNts name, std::string value){
		Element elem = { T_REGEX, STD_MOVE(name) };
		elem.large.swap(value);
		m_queue.push_back(STD_MOVE(elem));
	}
	void BsonBuilder::append_object(SharedNts name, const BsonBuilder &value){
		Element elem = { T_OBJECT, STD_MOVE(name) };
		value.build(elem.large);
		m_queue.push_back(STD_MOVE(elem));
	}
	void BsonBuilder::append_array(SharedNts name, const BsonBuilder &value){
		Element elem = { T_ARRAY, STD_MOVE(name) };
		value.build(elem.large);
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
			case T_OID:
				if(!::bson_append_oid(b.get(), it->name.get(), reinterpret_cast<const ::guint8 *>(it->small))){
					DEBUG_THROW(ProtocolException,
						sslit("Failed to append data to BSON object: bson_append_oid() failed"), -1);
				}
				break;

			case T_BOOLEAN:
				if(!::bson_append_boolean(b.get(), it->name.get(), reinterpret_cast<const bool &>(it->small))){
					DEBUG_THROW(ProtocolException,
						sslit("Failed to append data to BSON object: bson_append_boolean() failed"), -1);
				}
				break;

			case T_SIGNED:
				if(!::bson_append_int64(b.get(), it->name.get(), reinterpret_cast<const ::gint64 &>(it->small))){
					DEBUG_THROW(ProtocolException,
						sslit("Failed to append data to BSON object: bson_append_int64() failed"), -1);
				}
				break;

			case T_UNSIGNED:
				if(!::bson_append_int64(b.get(), it->name.get(), static_cast< ::gint64>(reinterpret_cast<const boost::uint64_t &>(it->small)))){
					DEBUG_THROW(ProtocolException,
						sslit("Failed to append data to BSON object: bson_append_int64() failed"), -1);
				}
				break;

			case T_DOUBLE:
				if(!::bson_append_double(b.get(), it->name.get(), reinterpret_cast<const double &>(it->small))){
					DEBUG_THROW(ProtocolException,
						sslit("Failed to append data to BSON object: bson_append_double() failed"), -1);
				}
				break;

			case T_STRING:
				if(it->large.size() > 0x7FFFFFFF){
					DEBUG_THROW(ProtocolException,
						sslit("Failed to append data to BSON object: string is too long"), -1);
				}
				if(!::bson_append_string(b.get(), it->name.get(), it->large.data(), static_cast< ::gint32>(it->large.size()))){
					DEBUG_THROW(ProtocolException,
						sslit("Failed to append data to BSON object: bson_append_string() failed"), -1);
				}
				break;

			case T_DATETIME:
				if(!::bson_append_utc_datetime(b.get(), it->name.get(),
					static_cast< ::gint64>(reinterpret_cast<const boost::uint64_t &>(it->small))))
				{
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
				if(it->large.size() > 0x7FFFFFFF){
					DEBUG_THROW(ProtocolException,
						sslit("Failed to append data to BSON object: blob is too long"), -1);
				}
				if(!::bson_append_binary(b.get(), it->name.get(), BSON_BINARY_SUBTYPE_USER_DEFINED,
					reinterpret_cast<const ::guint8 *>(it->large.data()), static_cast< ::gint32>(it->large.size())))
				{
					DEBUG_THROW(ProtocolException,
						sslit("Failed to append data to BSON object: bson_append_binary() failed"), -1);
				}
				break;

			case T_REGEX:
				if(!::bson_append_regex(b.get(), it->name.get(), it->large.c_str(), "")){
					DEBUG_THROW(ProtocolException,
						sslit("Failed to append data to BSON object: bson_append_regex() failed"), -1);
				}
				break;

			case T_OBJECT:
				if(it->large.size() > 0x7FFFFFFF){
					DEBUG_THROW(ProtocolException,
						sslit("Failed to create BSON object: BSON size is too large"), -1);
				}
				if(!temp.reset(::bson_new_from_data(reinterpret_cast<const ::guint8 *>(it->large.data()),
					static_cast< ::gint32>(std::max<std::size_t>(it->large.size(), 1) - 1))))
				{
					DEBUG_THROW(ProtocolException,
						sslit("Failed to create BSON object: bson_new() failed"), -1);
				}
				if(!::bson_append_document(b.get(), it->name.get(), temp.get())){
					DEBUG_THROW(ProtocolException,
						sslit("Failed to append data to BSON object: bson_append_document() failed"), -1);
				}
				break;

			case T_ARRAY:
				if(it->large.size() > 0x7FFFFFFF){
					DEBUG_THROW(ProtocolException,
						sslit("Failed to create BSON object: BSON size is too large"), -1);
				}
				if(!temp.reset(::bson_new_from_data(reinterpret_cast<const ::guint8 *>(it->large.data()),
					static_cast< ::gint32>(std::max<std::size_t>(it->large.size(), 1) - 1))))
				{
					DEBUG_THROW(ProtocolException,
						sslit("Failed to create BSON object: bson_new() failed"), -1);
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
